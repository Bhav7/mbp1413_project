from __future__ import print_function
import os

os.environ['MPLCONFIGDIR'] = '/home/zachvav/scratch/matplotlib_tmp'
#disable matplotlib to screen
import matplotlib.pyplot as plt
plt.switch_backend('agg')

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from models import *

import torch
import torch.optim

# from skimage.measure import compare_psnr
from utils.denoising_utils import *

def compare_psnr(img1, img2, normalize=False):
    if normalize:
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    mse = np.mean((img1 - img2)**2)
    return 10 * np.log10(1/mse)


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = True
sigma = 0.03 * 255
sigma_ = sigma/255.

for scan in os.listdir('../test'):

    for file in os.listdir(os.path.join('../test', scan)):
        if 'axial' in file and file.endswith('.png'):
            fname = os.path.join('../test', scan, file)
            print(fname)

            out_dir = os.path.join('out', scan)
            os.makedirs(out_dir, exist_ok=True)

            #set random seed for reproducibility
            torch.manual_seed(0)
            np.random.seed(0)

            img_pil = get_image(fname, imsize)[0]
            #add channel dimension to PIL
            img_pil = img_pil.convert('L')
            #crop top 25% and bottom 25% of image
            img_pil = img_pil.crop((0, int(160), img_pil.size[0], int(480)))
            img_np = pil_to_np(img_pil)

            print('img_pil.size', img_pil.size)
            print('img_np.shape', img_np.shape)

            img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_, clip=False)

            print('img_noisy_np.shape', img_noisy_np.shape)
            #min, max, mean, std
            print('img_noisy_np min', img_noisy_np.min())
            print('img_noisy_np max', img_noisy_np.max())
            print('img_noisy_np mean', img_noisy_np.mean())
            print('img_noisy_np std', img_noisy_np.std())

            if PLOT:
                plot_image_grid([img_np, img_noisy_np], 4, 6, save=True, save_path=os.path.join(out_dir, 'base.png'))


            #=========================================================

            INPUT = 'noise' # 'meshgrid'
            pad = 'reflection'
            OPT_OVER = 'net' # 'net,input'

            reg_noise_std = 1./30. # set to 1./20. for sigma=50
            LR = 0.01

            OPTIMIZER='adam' # 'LBFGS'
            show_every = 100
            exp_weight=0.99


            num_iter = 3000
            input_depth = 32
            figsize = 4 
            
            
            net = get_net(input_depth, 'skip', pad, n_channels=1,
                        skip_n33d=128, 
                        skip_n33u=128, 
                        skip_n11=4, 
                        num_scales=5,
                        upsample_mode='bilinear').type(dtype)
                
            net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

            # Compute number of parameters
            s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
            print ('Number of params: %d' % s)

            # Loss
            mse = torch.nn.MSELoss().type(dtype)

            img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

            #=========================================================

            base_psnr = compare_psnr(img_np, img_noisy_np)
            print('Base PSNR: ', base_psnr)

            net_input_saved = net_input.detach().clone()
            noise = net_input.detach().clone()
            out_avg = None
            last_net = None
            psrn_noisy_last = 0

            psnr_clean_history = []
            psnr_noised_history = []
            psnr_sm_clean_history = []

            i = 0
            def closure():
                
                global i, out_avg, psrn_noisy_last, last_net, net_input, psnr_clean_history, psnr_noised_history, base_psnr, psrn_sm_clean_history
                
                if reg_noise_std > 0:
                    net_input = net_input_saved + (noise.normal_() * reg_noise_std)
                
                out = net(net_input)
                
                # Smoothing
                if out_avg is None:
                    out_avg = out.detach()
                else:
                    out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
                        
                total_loss = mse(out, img_noisy_torch)
                total_loss.backward()
                    
                
                psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
                psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
                psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 

                psnr_clean_history.append(psrn_gt)
                psnr_noised_history.append(psrn_noisy)
                psnr_sm_clean_history.append(psrn_gt_sm)

                # Note that we do not have GT for the "snail" example
                # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
                print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
                if  PLOT and i % show_every == 0:
                    out_np = torch_to_np(out)
                    out_avg_np = torch_to_np(out_avg)
                    # plot_image_grid([np.clip(out_np, 0, 1), 
                    #                  np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1, save=True,
                    #                  save_path=os.path.join(out_dir, 'iter_%05d.png' % i))
                    
                    fig, axes = plt.subplots(1,5, figsize=(24,6))

                    #set plot title
                    fig.suptitle("Step %d    Noised → Clean PSNR %.2f" % (i, base_psnr), fontsize=16)
                    
                    #plot PSNR histroy in subplot 0,0
                    axes[0].plot(psnr_clean_history, label='PSNR (Clean)', color='orange')
                    axes[0].plot(psnr_sm_clean_history, label='PSNR (Clean) + SM', color='green')
                    axes[0].plot(psnr_noised_history, label='PSNR (Noised)', color='blue')
                    axes[0].set_xlabel('Step')
                    axes[0].set_ylabel('PSNR')
                    axes[0].set_title('PSNR History')
                    axes[0].legend()

                    #add dashed horizontal line to show PSNR of base image
                    axes[0].axhline(y=base_psnr, color='r', linestyle='--', label='Base PSNR')

                    #show output image
                    axes[1].imshow(out_np[0], cmap='gray')
                    axes[1].set_title('Model Output')
                    axes[1].axis('off')

                    #show avg output image
                    axes[2].imshow(out_avg_np[0], cmap='gray')
                    axes[2].set_title('Model Output (Smoothed)')
                    axes[2].axis('off')

                    #show gt image
                    axes[3].imshow(img_np[0], cmap='gray')
                    axes[3].set_title("Output → Clean PSNR: %.2f" % psrn_gt)
                    axes[3].axis('off')

                    #show noisy image
                    axes[4].imshow(img_noisy_np[0], cmap='gray')
                    axes[4].set_title('Output → Noised PSNR: %.2f' % psrn_noisy) 
                    axes[4].axis('off')

                    plt.savefig(os.path.join(out_dir, 'iter_%05d.png' % i), bbox_inches='tight')
                    plt.close()

                    #save output as numpy
                    np.save(os.path.join(out_dir, 'iter_%05d_output.npy' % i), out_np[0])

                    #save output as png
                    out_pil = np_to_pil(out_np)
                    out_pil.save(os.path.join(out_dir, 'iter_%05d_output.png' % i))

                    #save smoothed output as numpy
                    np.save(os.path.join(out_dir, 'iter_%05d_output_smoothoed.npy' % i), out_avg_np[0])

                    #save smoothed output as png
                    out_avg_pil = np_to_pil(out_avg_np)
                    out_avg_pil.save(os.path.join(out_dir, 'iter_%05d_output_smoothed.png' % i))

                    
                    
                    
                
                # Backtracking
                if i % show_every:
                    if psrn_noisy - psrn_noisy_last < -5: 
                        print('Falling back to previous checkpoint.')

                        for new_param, net_param in zip(last_net, net.parameters()):
                            net_param.data.copy_(new_param.cuda())

                        return total_loss*0
                    else:
                        last_net = [x.detach().cpu() for x in net.parameters()]
                        psrn_noisy_last = psrn_noisy
                        
                i += 1

                return total_loss

            p = get_params(OPT_OVER, net, net_input)
            optimize(OPTIMIZER, p, closure, LR, num_iter)

            out_np = torch_to_np(net(net_input))
            q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13, save=True,
                                save_path=os.path.join(out_dir, 'final.png'))