# PlantSeg Linux Server Delivery

This repository is adapted for a single RTX 4090 Linux server with Python 3.10, CUDA 11.8, and Miniconda.

## Directory contract

- Keep this repository and the dataset folder as siblings.
- The dataset must stay at `../plantseg` relative to the repository root.
- The dataset layout is expected to include:
  - `main.json`
  - `images/`
  - `ann/`

## Environment source of truth

- Use `environment.server.yml` as the only environment definition artifact.
- The environment pins PyTorch 2.0.0 CUDA 11.8 and `mmcv-full==1.7.2`.
- `opencv-python-headless` is installed before `mmcv-full` on purpose, so that MMCV does not pull the GUI OpenCV build.

## Automatic asset handling

- `google-bert/bert-base-uncased` is downloaded automatically into `checkpoints/bert-base-uncased/` when missing.
- The official Swin base checkpoint is downloaded automatically into `checkpoints/swin-base/` when missing.

## Training contract

- Dataset: `plantseg`
- Caption source: `caption[3]`
- Epochs: `50`
- Best checkpoint metric: foreground `IoU`
- Best checkpoint path: `logs/<exp>/ckpt.pth`
- Locked runtime defaults in `main.py` automatically enable `use_mask` and `use_pixel_decoder`, disable `use_exist`, switch to the PlantSeg config, and set the training batch size to `6` when the user keeps the default batch size.

## Evaluation contract

- Validation split is used during training.
- Final testing runs on the `test` split.
- Reported metrics are:
  - `IoU`
  - `Dice`
  - `Recall`
  - `mIoU`
  - `mACC`
- Test masks are written to `outputs/test_masks/<exp>/test/` and keep the same relative PNG paths as the ground-truth masks.

## Supported command surface

- Environment creation:
  - `conda env create -f environment.server.yml`
- Training:
  - `python main.py --dataset plantseg --exp plantseg_refsegformer`
- Evaluate an existing checkpoint on validation:
  - `python main.py --dataset plantseg --exp plantseg_refsegformer --eval --type val`
- Evaluate an existing checkpoint on test and export masks:
  - `python main.py --dataset plantseg --exp plantseg_refsegformer --eval --type test --save_masks`

## If OpenCV import fails on a server

- If training fails with `ImportError: libGL.so.1`, the active environment most likely resolved to `opencv-python` instead of the headless build.
- The intended fix is:
  - uninstall `opencv-python`
  - keep or reinstall `opencv-python-headless`
  - reinstall `mmcv-full` if needed after the OpenCV package is corrected
