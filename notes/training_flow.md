# PyTorch Training Flow Review Notes

Use this note as your reading guide before adding DDP.

## 1. Data path

`train.py` calls:

```text
build_dataloaders -> build_datasets -> build_transforms -> DataLoader
```

Important ideas:

- `Dataset` defines how to get one sample.
- `DataLoader` batches samples and may use worker processes.
- `pin_memory=True` can make CPU to GPU transfer faster.
- `non_blocking=True` only matters when memory is pinned and transfer can overlap.

## 2. Model path

`train.py` calls:

```text
build_model -> torchvision.models.resnet18 -> CIFAR stem modification
```

Why modify ResNet?

- ImageNet images are large.
- CIFAR images are 32x32.
- A 7x7 stride-2 conv plus maxpool is too aggressive for CIFAR.

## 3. Training step

One iteration does:

```text
load batch
move batch to GPU
forward
compute loss
zero gradients
backward
optimizer step
log metrics
```

This order matters.

## 4. Eval step

Evaluation uses:

```python
model.eval()
torch.no_grad()
```

This avoids training-specific behavior and saves memory.

## 5. Performance metrics to watch now

- `train_img_s`: images per second
- `data_time`: time spent waiting for dataloader
- `batch_time`: total batch iteration time

Later, in DDP, these become the baseline for scaling efficiency.
