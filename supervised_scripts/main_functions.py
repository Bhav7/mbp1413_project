import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

def get_data(path):
    """ 
    Get images from appropriate data split
    """
    to_use = []
    dir_present = os.listdir(f"data/{path}")
    for dir in dir_present:
        images = os.listdir(f"data/{path}/" + dir)
        for image in images:
            if "axial" in image:
                img_arr = plt.imread(f"data/{path}/{dir}/" + image)
                #ensure size is 320x320
                if img_arr.shape[0] > 320:
                    img_height = int((img_arr.shape[0] - 320)/2)
                    img_arr = img_arr[img_height:-img_height,:,:]
                if img_arr.shape[0] < 320 or img_arr.shape[1] < 320:
                    img_arr = cv2.resize(img_arr, (320, 320), interpolation=cv2.INTER_LINEAR)
                to_use.append(img_arr)
    return np.array(to_use)   

def calc_psnr(img1, img2):
    """
    Compute peak signal-to-noise ratio between two images
    """
    diff = img1 - img2
    mse = np.mean((diff)**2)
    return 10 * np.log10(1/mse)

def save_image(img, test_i, name, input_type = "torch"):
    """
    Save input array as image
    """
    if input_type == "torch":
        img = img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    img = img.reshape(320,320) if img.shape[2] == 1 else img
    img = np.clip(img, 0, 1) #needed to represent noise appropriately
    img = (img * 255).astype(np.uint8)
    img_pil = Image.fromarray(img)
    img_pil.save(f"results/outputs/test_{test_i}_{name}.png")