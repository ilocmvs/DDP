#!/usr/bin/env bash
set -euo pipefail

python train.py --config configs/cifar10_resnet18.yaml
