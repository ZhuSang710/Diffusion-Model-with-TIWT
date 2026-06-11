# Fast Diffusion Model for Image Denoising Based on Translation-Invariant Wavelet Transform



## Description
We propose a fast diffusion model for image denoising based on translation-invariant wavelet transform (TIWT). The method integrates three key components: 
(1) a TIWT-based feature enhancement module that performs multi-scale non-subsampled decomposition to strengthen the representation of image edges and textures, alleviating detail loss and textural distortion; 
(2) a non-uniform time-step sampling strategy that concentrates more steps on the most critical noise stages, optimizing the training and inference process; 
and (3) a spatial-channel attention mechanism (CBAM) embedded in the U-Net skip connections to ensure a more lightweight network structure. Compared with DDPM, the proposed method improves the peak signal-to-noise ratio (PSNR) by 1.21% and achieves substantially superior detail preservation.

The code is only for research purposes. If you have any questions regarding how to use this code, feel free to contact Dong Ran(15689457582@163.com).

## Publicly available Dataset
- Prostate-MRI-US-Biopsy dataset
- LDCT-and-Projection-data dataset
- BraTS 2018 dataset


## Code structure
* Please download from the official websites.
* After downloading, extract the file and put it into folder "data/". The directory structure should be as follows:

```bash
├── configs
│
├── data
│	├── LD_FD_CT_train
│	├── LD_FD_CT_test
│	├── PMUB-train
│	├── PMUB-test
│	├── Brats_train
│	└── Brats_test
│
├── datasets
│
├── functions
│
├── models
│
└── runners

```

## Usage
### 1. Git clone or download the codes.

### 2. Pretrained model weights
* We provide pretrained model weights for all tasks, where you can access them here:https://pan.baidu.com/s/1AXpDKtDRqptSdIJSU6JSXg?pwd=1234  .
* Pretrained model weights are also available on:https://pan.baidu.com/s/1AXpDKtDRqptSdIJSU6JSXg?pwd=1234. 




### 4. Training/Sampling a Fast-DDPM model
* Please make sure that the hyperparameters such as scheduler type and timesteps are consistent between training and sampling.
* The total number of time steps is defaulted as 1000 in the paper, so the number of involved time steps for Fast-DDPM should be less than 1000 as an integer.
```
python fast_ddpm_main.py --config {DATASET}.yml --dataset {DATASET_NAME} --exp {PROJECT_PATH} --doc {MODEL_NAME} --scheduler_type {SAMPLING STRATEGY} --timesteps {STEPS}
```
```
python fast_ddpm_main.py --config {DATASET}.yml --dataset {DATASET_NAME} --exp {PROJECT_PATH} --doc {MODEL_NAME} --sample --fid --scheduler_type {SAMPLING STRATEGY} --timesteps {STEPS}
```

where 
- `DATASET_NAME` should be selected among `LDFDCT` for image denoising task, `BRATS` for image-to-image translation task and `PMUB` for multi image super-resolution task.
- `SAMPLING STRATEGY` controls the scheduler sampling strategy proposed in the paper (either uniform or non-uniform).
- `STEPS` controls how many timesteps used in the training and inference process. It should be an integer less than 1000 for Fast-DDPM-TIWT, which is 10 by default.


### 5. Training/Sampling a DDPM model
* Please make sure that the hyperparameters such as scheduler type and timesteps are consistent between training and sampling.
* The total number of time steps is defaulted as 1000 in the paper, so the number of time steps for DDPM is defaulted as 1000.
```
python ddpm_main.py --config {DATASET}.yml --dataset {DATASET_NAME} --exp {PROJECT_PATH} --doc {MODEL_NAME} --timesteps {STEPS}
```
```
python ddpm_main.py --config {DATASET}.yml --dataset {DATASET_NAME} --exp {PROJECT_PATH} --doc {MODEL_NAME} --sample --fid --timesteps {STEPS}
```

where 
- `DATASET_NAME` should be selected among `LDFDCT` for image denoising task, `BRATS` for image-to-image translation task and `PMUB` for multi image super-resolution task.
- `STEPS` controls how many timesteps used in the training and inference process. It should be 1000 in the setting of this paper.

## Requirements
* Python==3.10.6
* torch==1.12.1
* torchvision==0.15.2
* numpy
* opencv-python
* tqdm
* tensorboard
* tensorboardX
* scikit-image
* medpy
* pillow
* scipy
* PyWavelets
* `pip install -r requirements.txt`

## References
* The code is mainly adapted from [DDIM](https://github.com/ermongroup/ddim).


