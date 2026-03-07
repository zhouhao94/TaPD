# TaPD: Temporal-adaptive Progressive Distillation for Observation-Adaptive Trajectory Forecasting in Autonomous Driving
### [[Paper]]()

## 🛠️ Get started

### Set up a new virtual environment
```
conda create -n TaPD python=3.10
conda activate TaPD
```

### Install dependency packages
```
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r ./requirements.txt
pip install av2==0.2.1
```

### Install Mamba
- We follow the settings outlined in [VideoMamba](https://github.com/OpenGVLab/VideoMamba).
```
git clone git@github.com:OpenGVLab/VideoMamba.git
cd VideoMamba
pip install -e causal-conv1d
pip install -e mamba
```

### Some packages may be useful
```
pip install tensorboard
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cu118.html
pip install protobuf==3.20.3
```

## 🕹️ Prepare the data
### Setup [Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html)
```
data_root
    ├── train
    │   ├── 0000b0f9-99f9-4a1f-a231-5be9e4c523f7
    │   ├── 0000b6ab-e100-4f6b-aee8-b520b57c0530
    │   ├── ...
    ├── val
    │   ├── 00010486-9a07-48ae-b493-cf4545855937
    │   ├── 00062a32-8d6d-4449-9948-6fedac67bfcd
    │   ├── ...
    ├── test
    │   ├── 0000b329-f890-4c2b-93f2-7e2413d4ca5b
    │   ├── 0008c251-e9b0-4708-b762-b15cb6effc27
    │   ├── ...
```

### Preprocess
```
python preprocess_av2.py --data_root=/path/to/data_root -p
```

### The structure of the dataset after processing
```
└── data
    └── TaPD_processed
        ├── train
        ├── val
        └── test
```

## 🔥 Training and testing

### Stage 1: Pre-train OAF
- **Step 1:** In `conf/config.yaml`, set `isFinetune` to `false` (line 12).
- **Step 2:** In `conf/model/model_forecast.yaml`, set `target._target_` to `src.model.trainer_forecast_av2_OAF.Trainer` (line 5).
- **Step 3:** In `src/datamodule/av2_datamodule.py`, set line 5 to: `from .av2_dataset import Av2Dataset, collate_fn`.
- **Step 4:** In `src/metrics/min_fde.py`, set lines 32 and 47 to: `fde=torch.norm(pred[...,-1,:2]-target.unsqueeze(1)[...,-1,:2],p=2,dim=-1)`.
- **Step 5:** Training OAF with `python train.py` and validation OAF with `python eval.py`.
- **Step 6:** After validation, save the checkpoint with the best validation result to `OAF.ckpt`.

### Stage 2: Train TBM Independently
- **Step 1:** In `conf/model/model_forecast.yaml`, set `target._target_` to `src.model.trainer_forecast_av2_TBM.Trainer` (line 5).
- **Step 2:** In `src/datamodule/av2_datamodule.py`, set line 5 to: `from .av2_dataset_TBM import Av2Dataset, collate_fn`.
- **Step 3:** In `src/metrics/min_fde.py`, set lines 32 and 47 to: `fde=torch.norm(pred[...,0,:2]-target.unsqueeze(1)[...,0,:2],p=2,dim=-1)`.
- **Step 4:** Training TBM with `python train.py` and validation TBM with `python eval.py`.
- **Step 5:** After validation, save the checkpoint with the best result to `TBM.ckpt`.

### Stage 3: Freeze TBM and Fine-tune OAF
- **Step 1:** In `conf/config.yaml`, set `isFinetune` to `true` (line 12), `pretrained_weights` to `OAF.ckpt` (line 16), and `backtrack_weights` to `TBM.ckpt` (line 17).
- **Step 2:** Reproduce Steps 2, 3, and 4 in Stage 1.
- **Step 3:** Finetune OAF with `python train.py`, and validation OAF with `python eval.py`.
- **Step 4:** After validation, save the checkpoint with the best result to `TaPD.ckpt`.
- **Step 5:** Test finetuned model for leaderboard submission with `python eval.py gpus=1 test=true`.

### Note: Validation under Variable-Length Observation
In `conf/datamodule/av2.yaml`, set `val_squence_start` to `0`, `10`, `20`, `30`, or `40` (line 18) to validate with observation length `50`, `40`, `30`, `20`, or `10`, respectively.

## ⭐ Results and checkpoints
| Models | 10Ts | 20Ts | 20Ts | 40Ts | 50Ts |
| :- | :-: | :-: | :-: | :-: | :-: |
| TaPD |  0.617/1.203  |  0.603/1.167  |  0.599/1.157  |  0.599/1.155  | 0.599/1.153 |

| Checkpoint | OAF | TBM | TaPD |
| :-: | :-: | :-: | :-: |
| -- | [OAF.ckpt](https://drive.google.com/file/d/1MADALJFCTv4wVCbNqvRjKcprnGcuFjsm/view?usp=drive_link) | [TBM.ckpt](https://drive.google.com/file/d/1s69_Y77G34chf7D8kmh6pn7ltLuD9-L9/view?usp=drive_link) | [TaPD.ckpt](https://drive.google.com/file/d/1g7dBDPnng0O-6ZLC57IFVrRkgdI4xfas/view?usp=drive_link) |

## ❤️ Acknowledgements
 - [VideoMamba](https://github.com/OpenGVLab/VideoMamba)
 - [Forecast-MAE](https://github.com/jchengai/forecast-mae)
 - [StreamPETR](https://github.com/exiawsh/StreamPETR)
 - [DeMo](https://github.com/fudan-zvg/DeMo)
