# Fast Diffusion Model for Image Denoising Based on Translation-Invariant Wavelet Transform



We propose a fast diffusion model for image denoising based on translation-invariant wavelet transform (TIWT). The method integrates three key components: (1) a TIWT-based feature enhancement module that performs multi-scale non-subsampled decomposition to strengthen the representation of image edges and textures, alleviating detail loss and textural distortion; (2) a non-uniform time-step sampling strategy that concentrates more steps on the most critical noise stages, optimizing the training and inference process; and (3) a spatial-channel attention mechanism (CBAM) embedded in the U-Net skip connections to ensure a more lightweight network structure. Compared with DDPM, the proposed method improves the peak signal-to-noise ratio (PSNR) by 1.21% and achieves substantially
superior detail preservation.


The code is only for research purposes. If you have any questions regarding how to use this code, feel free to contact Dong Ran(15689457582@163.com).

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
* `pip install -r requirements.txt`

## Publicly Available Datasets
- Prostate-MRI-US-Biopsy dataset
- LDCT-and-Projection-data dataset
- BraTS 2018 dataset
- The processed dataset can be accessed here: https://drive.google.com/file/d/1kF0g8fMR5XPQ2FTbutfTQ-hwG_mTqerx/view?usp=drive_link.

## Usage
### 1. Git clone or download the codes.

### 2. Pretrained model weights
* Model weights will be available soon.

### 3. Prepare data
* Please download our processed dataset or download from the official websites.
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

### 4. Training/Sampling the proposed fast diffusion model
* Please make sure that the hyperparameters such as scheduler type and timesteps are consistent between training and sampling.
* The standard DDPM uses 1000 time steps. The proposed method adopts a non-uniform sampling strategy to reduce the effective number of time steps. According to the paper, the optimal range is 10–50 steps, with 20 steps achieving the best trade-off between denoising quality and computational cost.
```
python fast_ddpm_main.py --config {DATASET}.yml --dataset {DATASET_NAME} --exp {PROJECT_PATH} --doc {MODEL_NAME} --scheduler_type {SAMPLING_STRATEGY} --timesteps {STEPS}
```
```
python fast_ddpm_main.py --config {DATASET}.yml --dataset {DATASET_NAME} --exp {PROJECT_PATH} --doc {MODEL_NAME} --sample --fid --scheduler_type {SAMPLING_STRATEGY} --timesteps {STEPS}
```

where 
- `DATASET_NAME` should be selected among `LDFDCT` for image denoising task, `BRATS` for image-to-image translation task, and `PMUB` for multi image super-resolution task.
- `SAMPLING_STRATEGY` controls the scheduler sampling strategy proposed in the paper: `non-uniform` (the proposed strategy) or `uniform`.
- `STEPS` controls how many timesteps are used in training and inference. The recommended range is 10–50, with a default of 10.


### 5. Training/Sampling a DDPM baseline
* Please make sure that the hyperparameters such as scheduler type and timesteps are consistent between training and sampling.
* The standard DDPM uses 1000 time steps as in the original paper.
```
python ddpm_main.py --config {DATASET}.yml --dataset {DATASET_NAME} --exp {PROJECT_PATH} --doc {MODEL_NAME} --timesteps {STEPS}
```
```
python ddpm_main.py --config {DATASET}.yml --dataset {DATASET_NAME} --exp {PROJECT_PATH} --doc {MODEL_NAME} --sample --fid --timesteps {STEPS}
```

where 
- `DATASET_NAME` should be selected among `LDFDCT` for image denoising task, `BRATS` for image-to-image translation task, and `PMUB` for multi image super-resolution task.
- `STEPS` controls how many timesteps are used in training and inference. It should be 1000 for the standard DDPM setting used in the paper.



