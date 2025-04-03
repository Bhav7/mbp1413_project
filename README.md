# mbp1413_project
Code for mbp1413 final project

## Supervised Scripts
 1) run_main_models.py
  - This script is responsible for both training the U-Net model and testing both the U-Net model and SwinIR
  - To use this code, besides the listed dependencies in the script, you need to clone the SwinIR repository and download [weights](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth) to the model
2) modify_final_layers.py
  - This script is responsible for both 1) finetuning the final conv layer of SwinIR with MRI data or 1) training a randomly init conv layer with MRI data. For computational reasons, train,val, and test images were ran through SwinIR and activations prior to the final layers were saved and used as input to these models. Unfortunately, these are too large to share, but can be recreated by running ```run_main_models.py --model_type swinir --extract_activations true ```
3) main_functions.py
  - This script has functions which are reused across other scripts
