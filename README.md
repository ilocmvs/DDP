# CIFAR ResNet Single-GPU Baseline

This is the first milestone of the flagship distributed training project: a clean single-GPU PyTorch image classifier on the CIFAR family.

The goal is not to chase maximum accuracy yet. The goal is to understand and own the full PyTorch training flow before adding DDP.

## What this project contains

- CIFAR-10 / CIFAR-100 dataset loading through `torchvision`
- ResNet-18 classifier adapted for CIFAR-sized images
- Single-GPU training and validation loops
- Config-driven experiment setup
- Metric logging for loss, accuracy, throughput, and epoch time
- TODO markers in key training-flow locations

## Project layout

```text
cifar_resnet_single_gpu/
  configs/
    cifar10_resnet18.yaml
  src/
    data.py        # dataset and dataloader creation
    model.py       # ResNet model factory
    engine.py      # train/eval loop logic
    utils.py       # config, metrics, checkpoint helpers
  train.py         # main entrypoint
  scripts/
    run_train.sh
  notes/
    training_flow.md
  requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install PyTorch using the official command for your CUDA version if the default pip install is not suitable:

https://pytorch.org/get-started/locally/

## Run

```bash
python train.py --config configs/cifar10_resnet18.yaml
```

or:

```bash
bash scripts/run_train.sh
```

## Suggested learning path

1. Read `train.py` first.
2. Read `src/data.py` to understand transforms and dataloaders.
3. Read `src/model.py` to see how ResNet is adapted for CIFAR.
4. Read `src/engine.py` carefully. This is the core training flow.
5. Fill TODOs one by one.
6. Confirm training loss decreases and validation accuracy improves.
7. Only after this baseline is clean, add DDP in the next milestone.

## Expected baseline

With ResNet-18 on CIFAR-10, this starter should train normally. Accuracy depends heavily on hyperparameters and number of epochs. For the initial learning pass, correctness and clean metrics matter more than final score.
