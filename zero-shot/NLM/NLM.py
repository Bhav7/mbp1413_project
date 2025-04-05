import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise
import os
import PIL

def compare_psnr(img1, img2, normalize=False):
    if normalize:
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    mse = np.mean((img1 - img2)**2)
    return 10 * np.log10(1/mse)

for scan in os.listdir('../test'):
    for file in os.listdir(os.path.join('../test', scan)):
        
        if 'axial' in file and file.endswith('.png'):
            fname = os.path.join('../test', scan, file)
            print(fname)

            #load MRI image
            img_path = fname
            img = PIL.Image.open(img_path)
            img = img.convert('L')
            img = img.crop((0, int(160), img.size[0], int(480)))
            img = np.array(img) / 255.0

            #add channel dimension to NP
            img = np.expand_dims(img, axis=2)

            astro = img

            print(astro.shape)

            sigma = 0.03

            #set random seed for reproducibility
            np.random.seed(0)
            noise = np.random.normal(0, sigma, astro.shape)
            noisy = astro + noise

            psnr_base = peak_signal_noise_ratio(astro, noisy)
            print(f'PSNR (base) = {psnr_base:0.2f}')

            # estimate the noise standard deviation from the noisy image
            sigma_est = np.mean(estimate_sigma(noisy, channel_axis=-1))
            print(f'estimated noise standard deviation = {sigma_est}')

            patch_kw = dict(
                patch_size=5,  # 5x5 patches
                patch_distance=6,  # 13x13 search area
                channel_axis=-1,
            )

            # slow algorithm
            denoise = denoise_nl_means(noisy, h=1.15 * sigma_est, fast_mode=False, **patch_kw)

            # slow algorithm, sigma provided
            denoise2 = denoise_nl_means(
                noisy, h=0.8 * sigma_est, sigma=sigma_est, fast_mode=False, **patch_kw
            )

            # fast algorithm
            denoise_fast = denoise_nl_means(noisy, h=0.8 * sigma_est, fast_mode=True, **patch_kw)

            # fast algorithm, sigma provided
            denoise2_fast = denoise_nl_means(
                noisy, h=0.6 * sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw
            )

            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 6), sharex=True, sharey=True)

            ax[0, 0].imshow(noisy, cmap='gray')
            ax[0, 0].axis('off')
            ax[0, 0].set_title('noisy')
            ax[0, 1].imshow(denoise, cmap='gray')
            ax[0, 1].axis('off')
            ax[0, 1].set_title('non-local means\n(slow)')
            ax[0, 2].imshow(denoise2, cmap='gray')
            ax[0, 2].axis('off')
            ax[0, 2].set_title('non-local means\n(slow, using $\\sigma_{est}$)')
            ax[1, 0].imshow(astro, cmap='gray')
            ax[1, 0].axis('off')
            ax[1, 0].set_title('original\n(noise free)')
            ax[1, 1].imshow(denoise_fast, cmap='gray')
            ax[1, 1].axis('off')
            ax[1, 1].set_title('non-local means\n(fast)')
            ax[1, 2].imshow(denoise2_fast, cmap='gray')
            ax[1, 2].axis('off')
            ax[1, 2].set_title('non-local means\n(fast, using $\\sigma_{est}$)')

            fig.tight_layout()

            #add channel dimension to denoise outputs
            denoise = np.expand_dims(denoise, axis=2)
            denoise2 = np.expand_dims(denoise2, axis=2)
            denoise_fast = np.expand_dims(denoise_fast, axis=2)
            denoise2_fast = np.expand_dims(denoise2_fast, axis=2)

            # print PSNR metric for each case
            psnr_noisy = compare_psnr(astro, noisy)
            psnr = compare_psnr(astro, denoise)
            psnr2 = compare_psnr(astro, denoise2)
            psnr_fast = compare_psnr(astro, denoise_fast)
            psnr2_fast = compare_psnr(astro, denoise2_fast)

            print(f'PSNR (noisy) = {psnr_noisy:0.2f}')
            print(f'PSNR (slow) = {psnr:0.2f}')
            print(f'PSNR (slow, using sigma) = {psnr2:0.2f}')
            print(f'PSNR (fast) = {psnr_fast:0.2f}')
            print(f'PSNR (fast, using sigma) = {psnr2_fast:0.2f}')

            # plt.show()
            plt.savefig(scan + '.png')

            #save PIL image
            out_pil = ((denoise2_fast - denoise2_fast.min()) / (denoise2_fast.max() - denoise2_fast.min()))
            out_pil = np.squeeze(out_pil, axis=2)
            out_pil = PIL.Image.fromarray((out_pil * 255).astype(np.uint8))
            out_pil.save(os.path.join(scan + '_output.png'))

            #save numpy of img
            out_np = np.squeeze(denoise2_fast, axis=2)
            np.save(os.path.join(scan + '_output.npy'), out_np)