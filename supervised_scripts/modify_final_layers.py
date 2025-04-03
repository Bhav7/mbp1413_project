import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from monai.networks.nets import UNet
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import argparse
import pickle

from SwinIR.models.network_swinir import SwinIR
from main_functions import calc_psnr, save_image, get_data

class conv_probe(nn.Module):
    def __init__(self, model_type, load_weights):
        super().__init__()
        if model_type == "swinir_probe":
            self.model = nn.Conv2d(180, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
    def forward(self, x):
        return self.model(x)
    

def get_activations(model_choice, path):
    """ 
    Get activations from layer prior to output for either SwinIR 
    """
    with open(f"results/probe_activations/{model_choice.split('_')[0]}/{model_choice.split('_')[0]}_{path}.pkl", "rb") as f:
        activations = pickle.load(f) 
    return activations["probe"]        

class Create_Dataset(Dataset):
    def __init__(self, input_data, output_data, model_choice, load_weights):
        self.input_data = input_data
        self.output_data = output_data.astype(np.float32)
        self.model_choice = model_choice
        self.load_weights = load_weights

    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        sample_input = self.input_data[idx]
        sample_output = self.output_data[idx][:,:,0]
        sample_output = np.expand_dims(sample_output, axis = 0)
 
        return sample_input.squeeze(0), sample_output

def predict(model, dataloader, device, model_type, mode = "test"):
    """ 
    Compute model performance on external dataset
    """
    model.eval()
    metrics_avg = []
    with torch.no_grad():
        for test_i, (noise_x, denoise_y) in enumerate(dataloader):
            output = model(noise_x.to(device))
            output = output[:, :, :denoise_y.shape[2], :denoise_y.shape[3]]
            psnr = calc_psnr(output.cpu().numpy(), denoise_y.cpu().numpy())
            metrics_avg.append(psnr)
            if mode == "test":
                save_image(output, test_i, model_type)
    return torch.mean(torch.tensor(metrics_avg))

def train(model, dataloader, optimizer, loss_func, device, model_choice):
    """
    Performes a single training step
    """
    model.train()
    loss_avg = []
    for noise_x, denoise_y in dataloader:
        optimizer.zero_grad()
        output = model(noise_x.to(device))
        output = output[:, :, :denoise_y.shape[2], :denoise_y.shape[3]]
        loss = loss_func(output, denoise_y.to(device))
        loss.backward()
        optimizer.step()
        loss_avg.append(loss.item())
    print(torch.mean(torch.tensor(loss_avg)))

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type = str, default = "swinir_probe")
    parser.add_argument("--load_weights", type = lambda x: x.lower() == "true", default = False)

    args = parser.parse_args()
    model_choice = args.model_type

    #LOAD DATA

    train_data, train_activations = get_data("train"), get_activations(model_choice, "train")
    val_data, val_activations = get_data("val"), get_activations(model_choice, "val")
    test_data, test_activations = get_data("test"), get_activations(model_choice, "test")    


    train_dataset = Create_Dataset(train_activations, train_data, model_choice, args.load_weights)
    train_loader = DataLoader(train_dataset, batch_size = 10, shuffle = True)

    val_dataset = Create_Dataset(val_activations, val_data, model_choice, args.load_weights)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = True)

    test_dataset = Create_Dataset(test_activations, test_data,  model_choice, args.load_weights)
    test_loader = DataLoader(test_dataset, batch_size = 1)

    device = "gpu" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using: {device}")

    
    model = conv_probe(model_choice, args.load_weights)

    #loads models final layer weights, if specified
    if args.load_weights:
        model.model.load_state_dict(torch.load(f"results/last_layer_weights/{model_choice.split('_')[0]}.pth"))

    model.to(device)

    #TRAIN MODEL
    loss_func = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    epochs = 1000

    print("Training")

    stop_counter = 0
    val_performance = []
    for epoch in range(epochs):
        train(model, train_loader, optimizer, loss_func, device, model_choice)
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

    #EVAL MODEL 
    print("Testing")
    test_score = predict(model, test_loader, device, model_choice)
    if args.load_weights:
        model_choice = model_choice + "ft_final"

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
