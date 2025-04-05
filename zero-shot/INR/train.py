import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from torch.autograd import Variable

from PIL import Image, ImageFilter
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
os.environ['MPLCONFIGDIR'] = '/home/zachvav/scratch/matplotlib_tmp'
#disable matplotlib to screen
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from skimage.metrics import peak_signal_noise_ratio\
from helpers import *

import time

#==================================================================

#loop over test dir
for scan in os.listdir('test'):
    for file in os.listdir(os.path.join('test', scan)):
        if 'axial' in file and file.endswith('.png'):

            out_dir = os.path.join('out', scan)
            os.makedirs(out_dir, exist_ok=True)

            #if out_dir/training.gif exists, skip this file
            if os.path.exists(os.path.join(out_dir, 'training.gif')):
                print("Skipping %s/%s" % (scan, file))
                continue
            
            img_path = os.path.join('test', scan, file)

            print(scan)

            sidelength = 320
            NOISE_STD = 0.03
            total_steps = 3000
            steps_til_summary = 10
            TV_weight = 1e-7
            # TV_weight = 0
            grad_weight = 0
            omega = 30.0
            exp_weight=0.99

            clean_img = get_mri_tensor(sidelength, img_path=img_path).cuda()
            with torch.no_grad():
                clean_img_gradient = calc_gradient(clean_img, shape=(sidelength, sidelength), blur=False)


            cameraman = ImageFitting(sidelength, noise_std=NOISE_STD, img_path=img_path)
            dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

            torch.manual_seed(5)

            noised_img = cameraman.pixels.cuda()
            noise_img_range = noised_img.cpu().view(sidelength,sidelength).detach().numpy().max() - noised_img.cpu().view(sidelength,sidelength).detach().numpy().min()
            with torch.no_grad():
                noised_img_gradient = calc_gradient(noised_img, shape=(sidelength, sidelength), blur=False)

            #save noised image to file
            noised_img_PIL = noised_img.cpu().view(sidelength,sidelength).detach().numpy()
            noised_img_PIL = (noised_img_PIL - noised_img_PIL.min()) / (noised_img_PIL.max() - noised_img_PIL.min())
            noised_img_PIL = (noised_img_PIL * 255).astype(np.uint8)
            noised_img_PIL = Image.fromarray(noised_img_PIL)
            noised_img_PIL = noised_img_PIL.convert('RGB')
            noised_img_PIL.save(os.path.join(out_dir, "noised_img.png"))


            img_siren = Siren(in_features=2, out_features=1, hidden_features=320, 
                            hidden_layers=3, outermost_linear=True, first_omega_0=omega, hidden_omega_0=omega)


            img_siren.cuda()

            #print number of parameters in the model
            print("Number of parameters in the model: %d" % sum(p.numel() for p in img_siren.parameters()))

            psnr_base = peak_signal_noise_ratio(clean_img.cpu().view(sidelength,sidelength).detach().numpy(),
                                noised_img.cpu().view(sidelength,sidelength).detach().numpy())
            print("PSNR (Base): %0.3f" % psnr_base)

            #add selective weight decay to the last two layers of the network
            optim = torch.optim.Adam(lr=1e-4, params=[
                {'params': img_siren.net[0].parameters(), 'weight_decay': 0.0},
                {'params': img_siren.net[1].parameters(), 'weight_decay': 0.0},
                {'params': img_siren.net[2].parameters(), 'weight_decay': 0.0},
                {'params': img_siren.net[3].parameters(), 'weight_decay': 0.001},
                {'params': img_siren.net[4].parameters(), 'weight_decay': 0.001}
            ])


            model_input, ground_truth = next(iter(dataloader))
            model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

            psnr_clean_history = []
            psnr_clean_avg_history = []
            psnr_noised_history = []
            entropy_history = []

            out_avg = None

            for step in range(total_steps):

                model_output, coords = img_siren(model_input)

                if out_avg is None:
                    out_avg = model_output.detach()
                else:
                    out_avg = out_avg * exp_weight + model_output.detach() * (1 - exp_weight)


                img_grad = calc_gradient(model_output, shape=(sidelength, sidelength))


                loss = ((model_output - ground_truth)**2).mean()
                
                loss_TV = compute_TV(model_output.view(sidelength, sidelength))     #add TV term
                loss_GRAD = ((img_grad - noised_img_gradient)**2).mean() #add gradient MSE loss

                # loss += loss_GRAD
                loss = (1 - grad_weight) * loss + grad_weight * loss_GRAD + TV_weight * loss_TV


                psnr_clean = peak_signal_noise_ratio(clean_img.cpu().view(sidelength,sidelength).detach().numpy(),
                                                     model_output.cpu().view(sidelength,sidelength).detach().numpy())
                psnr_clean_avg = peak_signal_noise_ratio(clean_img.cpu().view(sidelength,sidelength).detach().numpy(),
                                                         out_avg.cpu().view(sidelength,sidelength).numpy())
                psnr_noised = peak_signal_noise_ratio(noised_img.cpu().view(sidelength,sidelength).detach().numpy(),
                                                      model_output.cpu().view(sidelength,sidelength).detach().numpy(), data_range=noise_img_range)
                entropy = gradient_entropy(model_output, shape=(sidelength, sidelength))
                
                psnr_clean_history.append(psnr_clean)
                psnr_clean_avg_history.append(psnr_clean_avg)
                psnr_noised_history.append(psnr_noised)
                entropy_history.append(entropy.item())
                
                if not step % steps_til_summary:

                    print("Step %d, Total loss %0.6f loss (GRAD) %0.6f PSNR (Clean) %0.3f PSNR SM (Clean) %0.3f PSNR (Noised) %0.3f Gradient Entropy %0.5f" % (step, loss.item(), loss_GRAD.item(), psnr_clean, psnr_clean_avg, psnr_noised, entropy.item()))

                    fig, axes = plt.subplots(2,4, figsize=(24,12))

                    #set plot title
                    fig.suptitle("Step %d    Noised → Clean PSNR %.2f" % (step, psnr_base), fontsize=16)

                    #set axes titles
                    axes[0,0].set_title("PSNR History")
                    axes[0,1].set_title("Model Output")
                    axes[0,2].set_title("Output → Clean PSNR: %.2f" % psnr_clean)
                    axes[0,3].set_title("Output → Noised PSNR: %.2f" % psnr_noised)
                    axes[1,0].set_title("Gradient Entropy History")
                    axes[1,1].set_title("Output Gradient")
                    axes[1,2].set_title("Clean Gradient")
                    axes[1,3].set_title("Noised Gradient")

                    #set space between subplots
                    plt.subplots_adjust(wspace=0.05, hspace=0.1)

                    vmin = min(model_output.min(), clean_img.min(), noised_img.min()).item()
                    vmax = max(model_output.max(), clean_img.max(), noised_img.max()).item()
                    vmin_grad = min(img_grad.min(), clean_img_gradient.min(), noised_img_gradient.min()).item()
                    vmax_grad = max(img_grad.max(), clean_img_gradient.max(), noised_img_gradient.max()).item()

                    #plot PSNR histroy in subplot 0,0
                    axes[0,0].plot(psnr_clean_history, label='PSNR (Clean)', color='orange')
                    axes[0,0].plot(psnr_noised_history, label='PSNR (Noised)', color='blue')
                    axes[0,0].plot(psnr_clean_avg_history, label='PSNR (Clean) + SM', color='green')
                    axes[0,0].set_xlabel('Step')
                    axes[0,0].set_ylabel('PSNR')
                    axes[0,0].set_title('PSNR History')
                    axes[0,0].legend()

                    #add dashed horizontal line to show PSNR of base image
                    axes[0,0].axhline(y=psnr_base, color='r', linestyle='--', label='Base PSNR')

                    #plot gradient entropy history in subplot 1,0
                    axes[1,0].plot(entropy_history, label='Gradient Entropy', color='green')
                    axes[1,0].set_xlabel('Step')
                    axes[1,0].set_ylabel('Entropy')
                    axes[1,0].set_title('Gradient Entropy History')
                    axes[1,0].legend()

                    axes[0,1].imshow(model_output.cpu().view(sidelength,sidelength).detach().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
                    axes[0,2].imshow(clean_img.cpu().view(sidelength,sidelength).detach().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
                    axes[0,3].imshow(noised_img.cpu().view(sidelength,sidelength).detach().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
                    axes[1,1].imshow(img_grad.norm(dim=-1).cpu().view(sidelength,sidelength).detach().numpy(), cmap='hot', vmin=vmin_grad, vmax=vmax_grad)
                    axes[1,2].imshow(clean_img_gradient.norm(dim=-1).cpu().view(sidelength,sidelength).detach().numpy(), cmap='hot', vmin=vmin_grad, vmax=vmax_grad)
                    axes[1,3].imshow(noised_img_gradient.norm(dim=-1).cpu().view(sidelength,sidelength).detach().numpy(), cmap='hot', vmin=vmin_grad, vmax=vmax_grad)

                    #turn off axis for all subplots
                    axes[0,1].axis('off')
                    axes[0,2].axis('off')
                    axes[0,3].axis('off')
                    axes[1,1].axis('off')
                    axes[1,2].axis('off')
                    axes[1,3].axis('off')

                    #save plot
                    plt.savefig(os.path.join(out_dir, "%04d.png" % step))
                    plt.close(fig)

                    #save model output in single PNG file
                    model_out_img = model_output.cpu().view(sidelength,sidelength).detach().numpy()
                    model_out_img = (model_out_img - model_out_img.min()) / (model_out_img.max() - model_out_img.min())
                    model_out_img = (model_out_img * 255).astype(np.uint8)
                    model_out_img = Image.fromarray(model_out_img)
                    model_out_img = model_out_img.convert('RGB')
                    model_out_img.save(os.path.join(out_dir, "%04d_output.png" % step))

                    #save model output to numpy file
                    model_out_np = model_output.cpu().view(sidelength,sidelength).detach().numpy()
                    np.save(os.path.join(out_dir, "%04d_output.npy" % step), model_out_np)

                    #save smoothed model output in single PNG file
                    model_out_avg_img = out_avg.cpu().view(sidelength,sidelength).detach().numpy()
                    model_out_avg_img = (model_out_avg_img - model_out_avg_img.min()) / (model_out_avg_img.max() - model_out_avg_img.min())
                    model_out_avg_img = (model_out_avg_img * 255).astype(np.uint8)
                    model_out_avg_img = Image.fromarray(model_out_avg_img)
                    model_out_avg_img = model_out_avg_img.convert('RGB')
                    model_out_avg_img.save(os.path.join(out_dir, "%04d_output_avg.png" % step))

                    #save smoothed model output to numpy file
                    model_out_avg_np = out_avg.cpu().view(sidelength,sidelength).detach().numpy()
                    np.save(os.path.join(out_dir, "%04d_output_avg.npy" % step), model_out_avg_np)

                optim.zero_grad()
                loss.backward()
                optim.step()

            with torch.no_grad():
                #plot PSNR history
                plt.plot(psnr_clean_history, label='PSNR (Clean)', color='orange')
                plt.plot(psnr_noised_history, label='PSNR (Noised)', color='blue')
                plt.xlabel('Step')
                plt.ylabel('PSNR')
                plt.title('PSNR History')
                plt.legend()
                plt.savefig(os.path.join(out_dir, "psnr_history.png"))
                plt.close()

                #plot grad entropy history
                plt.plot(entropy_history, label='Gradient Entropy', color='green')
                plt.xlabel('Step')
                plt.ylabel('Entropy')
                plt.title('Gradient Entropy History')
                plt.legend()
                plt.savefig(os.path.join(out_dir, "entropy_history.png"))
                plt.close()

                #output super-resolution image
                upsample_factor = 4
                coords_superres = get_mgrid(sidelength*upsample_factor, 2).cuda()
                model_output_superres, _ = img_siren(coords_superres)
                model_output_superres = model_output_superres.view(sidelength*upsample_factor, sidelength*upsample_factor).cpu().detach().numpy()

                #get downscaled super-resolved image
                model_output_superres_down = Image.fromarray(model_output_superres)
                model_output_superres_down = model_output_superres_down.resize((sidelength, sidelength), Image.BICUBIC)
                model_output_superres_down = np.array(model_output_superres_down)

                #calc PSNR of super-resolved image
                psnr_superres = peak_signal_noise_ratio(clean_img.cpu().view(sidelength,sidelength).detach().numpy(), model_output_superres_down)
                print(f"PSNR (Super-resolved): {psnr_superres:.3f}")

                #save super-resolved image=======================================
                model_output_superres = (model_output_superres - model_output_superres.min()) / (model_output_superres.max() - model_output_superres.min())
                model_output_superres = (model_output_superres * 255).astype(np.uint8)
                model_output_superres = Image.fromarray(model_output_superres)
                model_output_superres = model_output_superres.convert('RGB')
                model_output_superres.save(os.path.join(out_dir, "superres.png"))

                #save down-scaled super-resolved image===========================
                model_output_superres_down = (model_output_superres_down - model_output_superres_down.min()) / (model_output_superres_down.max() - model_output_superres_down.min())
                model_output_superres_down = (model_output_superres_down * 255).astype(np.uint8)
                model_output_superres_down = Image.fromarray(model_output_superres_down)
                model_output_superres_down = model_output_superres_down.convert('RGB')
                model_output_superres_down.save(os.path.join(out_dir, "superres_down.png"))

                #save GIF of training images
                images = []

                for step in range(total_steps):
                    if step % steps_til_summary == 0:
                        img = Image.open("out_img/%04d.png" % step)
                        images.append(img)

                images[0].save(os.path.join(out_dir,'training.gif'), save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

                #save model
                torch.save(img_siren.state_dict(), os.path.join(out_dir, "model.pth"))