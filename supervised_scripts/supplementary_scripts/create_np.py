import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_data(path):
    to_use = []
    dir_present = os.listdir(f"data/{path}")
    for dir in dir_present:
        images = os.listdir(f"data/{path}/" + dir)
        for image in images:
            if "axial" in image:
                img_arr = plt.imread(f"data/{path}/{dir}/" + image)
                if img_arr.shape[0] > 320:
                    img_height = int((img_arr.shape[0] - 320)/2)
                    img_arr = img_arr[img_height:-img_height,:,:]
                if img_arr.shape[0] < 320 or img_arr.shape[1] < 320:
                    img_arr = cv2.resize(img_arr, (320, 320), interpolation=cv2.INTER_LINEAR)
                to_use.append(img_arr)
                # to_use.append(cv2.resize(img_arr, (320, 320), interpolation=cv2.INTER_LINEAR))
    return np.array(to_use)  

if __name__ == "__main__":

    train_data = get_data("train")
    val_data = get_data("val")
    test_data = get_data("test")

    np.save("/Users/bhavverma/Documents/coursework/mbp_ai/data/np_data/train_data/train_data.npy", train_data)
    np.save("/Users/bhavverma/Documents/coursework/mbp_ai/data/np_data/test_data/test_data.npy", test_data)
    np.save("/Users/bhavverma/Documents/coursework/mbp_ai/data/np_data/val_data/val_data.npy", val_data)