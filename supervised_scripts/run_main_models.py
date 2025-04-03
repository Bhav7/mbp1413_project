import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from monai.networks.nets import UNet
from monai.networks.layers import Norm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import argparse
import pickle

from SwinIR.models.network_swinir import SwinIR
from main_functions import calc_psnr, save_image, get_data

def noise(sample, seed  = 42):
    """ 
    Add guassian noise to image 
    """
    torch.manual_seed(seed)
    return (sample + (torch.randn_like(sample) * 0.03)).float()

class Create_Dataset(Dataset):
    def __init__(self, data, transform = None):
        self.data = data.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx][:,:,0]
        sample = torch.tensor(sample).unsqueeze(0)

        if self.transform:
            sample = self.transform(sample)

        return noise(sample), sample
    
def train(model, dataloader, optimizer, loss_func, device):
    """
    Performes a single training step
    """
    model.train()
    loss_avg = []
    for noise_x, denoise_y in dataloader:
        optimizer.zero_grad()
        output = model(noise_x.to(device))
        loss = loss_func(output, denoise_y.to(device))
        loss.backward()
        optimizer.step()
        loss_avg.append(loss.item())
        # with torch.no_grad():
        #     psnr = calc_psnr(output.cpu().detach().numpy(), denoise_y.detach().numpy())
        #     print(psnr)
    print(torch.mean(torch.tensor(loss_avg)))


features = {"probe":[]}
def hook_fn(module, input, output):
    """
    Get intermediate activations of model
    """
    features["probe"].append(output)

def SwinIR_format(noised, window_size = 8):
    """
    SwinIR pre-input formatting needed to correctly use image w/ model
    Sourced from: https://github.com/JingyunLiang/SwinIR/blob/main/predict.py
    """
    _, _, h_old, w_old = noised.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    noised = torch.cat([noised, torch.flip(noised, [2])], 2)[:, :, :h_old + h_pad, :]
    noised = torch.cat([noised, torch.flip(noised, [3])], 3)[:, :, :, :w_old + w_pad]
    return noised, h_old, w_old

def predict(model, dataloader, device, model_type, mode = "test"):
    """ 
    Compute model performance on external dataset
    """
    model.eval()
    metrics_avg = []
    with torch.no_grad():
        if model_type == "swinir":
            model.conv_after_body.register_forward_hook(hook_fn)
        for test_i, (og_noise_x, denoise_y) in enumerate(dataloader):
            if model_type == "swinir":
                    noise_x, h_old, w_old = SwinIR_format(og_noise_x)
            else:
                noise_x = og_noise_x
            output = model(noise_x.to(device))
            if model_type == "swinir":
                output = output[..., :h_old, :w_old]
            psnr = calc_psnr(output.cpu().detach().numpy(), denoise_y.detach().numpy())
            # print(calc_psnr(og_noise_x.cpu().detach().numpy(), denoise_y.detach().numpy()))
            metrics_avg.append(psnr)
            if mode == "test":
               save_image(output, test_i, model_type) 
            #    save_image(og_noise_x, test_i, "noise") 
            #    save_image(denoise_y, test_i, "truth") 
               
    #Bug, where can only save activations if treated as test; can fix at some point
    # with open("results/probe_activations/swinir/swinir_test.pkl", "wb") as f:
    #     pickle.dump(features, f)
    return torch.mean(torch.tensor(metrics_avg))

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type = str, default = "swinir")
    parser.add_argument("--data_percent", type = float, default = 1)

    args = parser.parse_args()
    model_choice = args.model_type
    data_percent = args.data_percent
    
    #LOAD DATA
    train_data = get_data("train")

    #useful for experimenting with data scaling
    amount_to_keep = int(len(train_data)*data_percent)
    train_data = train_data[:amount_to_keep]

    val_data = get_data("val")
    test_data = get_data("test")
    
    if model_choice == "swinir":
        bs = 1 #will not fit in memory otherwise
    else:
        bs = 32
 
    train_dataset = Create_Dataset(train_data, transform = transforms.Compose([transforms.RandomResizedCrop((320, 320)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)]))
    train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = True)


    val_dataset = Create_Dataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = True)

    test_dataset = Create_Dataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size = 1)

    # for noised, denoised in test_loader:
    #     for i in range(len(denoised)):
    #         noised_img = noised[i].permute(1,2,0)
    #         print(noised_img.shape)
    #         plt.imshow(noised_img, cmap='gray')
    #         plt.show()
    #         break


    device = "gpu" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using: {device}")

    #SELECT MODEL
    if model_choice == "swinir":
        model = SwinIR(upscale=1, in_chans=1, img_size=128, window_size=8,
    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2, upsampler='', resi_connection='1conv'
    )
        model.load_state_dict(torch.load('SwinIR/model_zoo/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth')['params'], strict=True)
        if not(os.path.exists("results/last_layer_weights/swinir.pth")):
            torch.save(model.conv_last.state_dict(), "results/last_layer_weights/swinir.pth")
        model.to(device)
        
    elif model_choice == "unet":
        epochs = 1000
        if os.path.exists("results/unet_hps.pkl"):
            with open(f"results/unet_hps.pkl", "rb") as f:
                hyperparameter_search_performances = pickle.load(f)
        else:
            #hp search
            hyperparameter_search_performances = []
            search_space = {"num_res_units": [1, 2, 3], "channels": [(3, 16, 32, 64, 128), (6, 32, 64, 128, 256)]}
            for res_idx, num_res_unit in enumerate(search_space["num_res_units"]):
                for channel_idx, channels in enumerate(search_space["channels"]):
                    model = UNet(
                        spatial_dims=2,
                        in_channels=1,
                        out_channels=1,
                        channels=channels,
                        strides=(2, 2, 2, 2),
                        num_res_units=num_res_unit,
                        norm=Norm.BATCH
                    ).to(device)

                    loss_func = nn.MSELoss()
                    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)

                    print(f"Training combination: num_res_units:{num_res_unit}, channels:{channels}")
                    stop_counter = 0
                    val_performance = []
                    for epoch in range(epochs):
                        train(model, train_loader, optimizer, loss_func, device)
                        val_performance.append(predict(model, val_loader, device, model_choice, mode = "val"))
                        if epoch == 0:
                            continue
                        else:
                            if val_performance[-1] > max(val_performance[:-1]):
                                stop_counter = 0
                            else:
                                stop_counter +=1
                        if stop_counter == 25: #will terminate training when psnr does not increase for 25 epochs on the val set
                            break

                    hyperparameter_search_performances.append((max(val_performance[:-1]), num_res_unit, channels))
                    print(max(val_performance[:-1]))

            with open(f"results/unet_hps.pkl", "wb") as f:
                pickle.dump(hyperparameter_search_performances, f)

        hyperparameter_search_performances.sort(key = lambda x: x[0])
        best_hps = hyperparameter_search_performances[-1]

        print(f"Training with best combination: num_res_units:{best_hps[1]}, channels:{best_hps[2]}")

        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=best_hps[2],
            strides=(2, 2, 2, 2),
            num_res_units=best_hps[1],
            norm=Norm.BATCH
        ).to(device)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)

        print("Training")
        stop_counter = 0
        val_performance = []
        for epoch in range(epochs):
            train(model, train_loader, optimizer, loss_func, device)
            val_performance.append(predict(model, val_loader, device, model_choice, mode = "val"))
            if epoch == 0:
                continue
            else:
                if val_performance[-1] > max(val_performance[:-1]):
                    stop_counter = 0
                else:
                    stop_counter +=1
            if stop_counter == 25: #will terminate training when psnr does not increase for 25 epochs on the val set
                break
        
        model_choice = model_choice + "_" + str(data_percent)

    #EVAL MODEL
    print("Testing")
    test_score = predict(model, test_loader, device, model_choice)

    if os.path.exists("results/supervised_results.pkl"):
        with open(f"results/supervised_results.pkl", "rb") as f:
            supervised_results = pickle.load(f)
        supervised_results[model_choice] = test_score.item()
    else:
        supervised_results = {}
        supervised_results[model_choice] = test_score.item()
    
    print(supervised_results)

    with open(f"results/supervised_results.pkl", "wb") as f:
        pickle.dump(supervised_results, f)


