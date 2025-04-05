# mbp1413_project
Code for mbp1413 final project

## Supervised Scripts
 1) run_main_models.py
  - This script is responsible for both 1) training the U-Net model and 2) testing both the U-Net model and SwinIR
  - To use this code, besides the listed dependencies in the script, you need to clone the SwinIR repository and download [weights](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth) to the model. Also you need to request access to use [fastMRI](https://fastmri.med.nyu.edu)
2) modify_final_layers.py
  - This script is responsible for both 1) finetuning the final conv layer of SwinIR with MRI data and 2) training a randomly init conv layer with MRI data. For computational reasons, train, val, and test images were ran through SwinIR and activations prior to the final layers were saved and used as input to these models. Unfortunately, these are too large to share, but can be recreated by running ```run_main_models.py --model_type swinir --extract_activations true ```
3) main_functions.py
  - This script has functions which are reused across other scripts
4) plots.ipynb
  - Notebook with code used to generate plots for supervised portion of paper
5) results
  - This directory stores the PSNR results for each of the supervised models in the paper alongside the hyperparameter search results for U-Net


## Zero-shot scripts
  1) DIP/DIP.py
   - This script trains a Deep Image Prior on all MRI PNG images present in ../test directory
   - It will output example images at specific steps and show PSNR over time plots to "out" directory
  2) INR/train.py
    - This script trains a SIREN implicit neural representation on all MRI PNG images present in ../test directory
    - will output example images as well as plots showing PSNR over time to "out" directory
  3) NLM/NLM.py
    - This script runs non-local means denoising algorithm on all MRI png images in ../test directory