# CU-CVF
The implementation follows the method described in **Closed-Loop Uncertainty-Guided Variational Fusion of VFM-CNN Representations for Coronary Artery Segmentation**.


## 1. Method Overview

The model consists of four main components:

1. **Parallel encoders**: SAM-Med3D extracts pretrained volumetric features, while a residual 3D CNN extracts local structural features. By default, only the final two ViT blocks of the SAM image encoder are unfrozen.
2. **Attention-Guided Enhancement (AGE)**: Spatial attention enhances the SAM output features, followed by a projection from 384 to 256 channels.
3. **Closed-Loop Uncertainty-Guided Cross-Branch Variational Fusion (CU-CVF)**: Gaussian feature posteriors and attention posteriors are learned for both branches. Latent representations are combined using Softmax weights normalized across the branch dimension.
4. **Evidential-Learning Uncertainty Refinement (EUR)**: Features from four decoder scales are aggregated. Dirichlet evidence provides voxel-wise uncertainty estimates that guide adaptive refinement of the segmentation logits.


## 2. Repository Structure

The minimal training layout is shown below. Pretrained weights are stored locally and do not need to be included in the code repository.

```text
.
├── README.md
├── train3d.py
├── model/
│   ├── __init__.py
│   ├── CUCVF3D_sam.py
│   └── SAM-Med3D-main/
│       └── ckpt/
│           └── sam_med3d_turbo.pth       # Downloaded pretrained weights
└── dataset_ASOCA/
    ├── folder1/
    │   ├── folder1_train.list
    │   ├── folder1_validation.list
    │   └── folder1_test.list
    ├── folder2/
    ├── folder3/
    ├── folder4/
    └── folder5/
        ├── folder5_train.list
        ├── folder5_validation.list
        └── folder5_test.list
```

- `model/CUCVF3D_sam.py`: The model, AGE, CU-CVF, EUR, and `CUCVFCompositeLoss`.
- `train3d.py`: Data loading, augmentation, optimization, training, sliding-window validation, logging, and full training checkpoints.
- `dataset_ASOCA/`: Case split lists. The actual NPY volumes may be stored outside the repository.


## 3. Installation and Pretrained Weights

### 3.1 Install Dependencies

Use a dedicated Python environment, for example:

```bash
conda create -n cucvf python=3.12
conda activate cucvf
```


Install the direct dependencies:

```bash
python -m pip install numpy scipy medim
```

The [MedIM project](https://github.com/uni-medical/MedIM) provides the `medim.create_model(...)` interface. The current NPY training entry point does not directly depend on torchvision, SimpleITK, or MONAI; MedIM's transitive dependencies are handled by the package installer.

Check the environment before training:

```bash
python -c "import torch, numpy, scipy, medim; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GradScaler:', hasattr(torch.amp, 'GradScaler'))"
```

### 3.2 Prepare SAM-Med3D Weights

Obtain [sam_med3d_turbo.pth](https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth) through the channels listed in the [official SAM-Med3D repository](https://github.com/uni-medical/SAM-Med3D), and place it at:

```text
model/SAM-Med3D-main/ckpt/sam_med3d_turbo.pth
```

Alternatively, store the weights elsewhere and specify their location with `--sam-checkpoint /path/to/sam_med3d_turbo.pth`.

The current model constructor checks for a local file. It does not accept a remote URL through `--sam-checkpoint` or automatically download a missing checkpoint.

## 4. Data Preparation

###  NPY Files

Organize the data as follows:

```text
/data/ASOCA/npy/
├── img/
│   ├── Normal_0.npy
│   ├── Normal_1.npy
│   ├── Normal_2.npy
│   └── Normal_3.npy
└── mask/
    ├── Normal_0.npy
    ├── Normal_1.npy
    ├── Normal_2.npy
    └── Normal_3.npy
```

Each image/mask pair must have matching filenames and satisfy the following requirements:



## 5. Training

Run the following commands from the repository root. Adjust data paths and GPU indices for your environment.

### 5.1 Default Training

```bash
python train3d.py --data-path /data/ASOCA/npy --folder folder5 --device cuda:0
```

Specify a different pretrained checkpoint location:

```bash
python train3d.py --data-path /data/ASOCA/npy --device cuda:0 --sam-checkpoint /checkpoints/sam_med3d_turbo.pth
```

### 5.2 JSON Configuration

You can create a `config.json` file, for example:

```json
{
  "data_path": "/data/ASOCA/npy",
  "folder": "folder5",
  "device": "cuda:0",
  "run_name": "asoca_fold5_poly",
  "epochs": 800,
  "batch_size": 2,
  "input_shape": [128, 160, 160],
  "lr": 0.001,
  "poly_power": 0.9,
  "weight_decay": 0.0,
  "class_weights": [0.2, 0.8],
  "lambda_ug": 0.0001,
  "evidence_annealing_epochs": 50,
  "max_grad_norm": 0.0,
  "amp": false
}
```

```bash
python train3d_asoca.py --config config.json
```


## 6. Default Hyperparameters

### 6.1 Optimization and Regularization

| Parameter | Default |
| --- | --- |
| Epochs / batch size / patch | `800` / `2` / `128×160×160` |
| Optimizer | Adam |
| Initial learning rate for all trainable parameters | `1e-3`, including the unfrozen SAM-Med3D blocks |
| Learning-rate schedule | Polynomial decay, `power=0.9` |
| Weight decay | `0.0`, disabled by default |
| WCE class weights: background / foreground | `0.2 / 0.8` |
| Fixed UG-KL weight | `1e-4`, active from the first epoch |
| Evidential Dirichlet KL annealing | `50` epochs |
| Dynamic prior `sigma0 / gamma / eps` | `1.0 / 2.0 / 1e-6` |
| Global gradient norm clipping | `0.0`, disabled by default |
| Random seed | `2026` |
| Default precision | FP32 |

The learning-rate schedule is:
```text
lr(t) = 1e-3 * (1 - t / T)^0.9
T = 800
```

The UG-KL coefficient and the evidential Dirichlet KL coefficient are distinct:
```text
lambda_UG(t) = 1e-4
lambda_evidence_KL(t) = min(1, t / Ta)
Ta = 50
```



## 7. Related Resources

- SAM-Med3D: [Official repository](https://github.com/uni-medical/SAM-Med3D).
- MedIM: [Model construction and installation](https://github.com/uni-medical/MedIM).
- PyTorch: [Installation guide](https://pytorch.org/get-started/locally/) and [AMP interface](https://docs.pytorch.org/docs/2.3/amp.html).
