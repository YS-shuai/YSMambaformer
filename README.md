# Mambaformer: Multi-Scale Fusion for Spatiotemporal Sequence Prediction
This repository is an open-source project implementing video prediction benchmarks using the **Mambaformer** model, which includes the implementation code corresponding to the manuscript. Currently, this project is intended solely for reviewers' reference.
## Introduction
<img src="[https://github.com/YouZhou12138/PredRNNv3/blob/main/imgs/DB_LSTM.png](https://github.com/YS-shuai/YSMambaformer/blob/main/imgs/Mambaformer.png)" style="display: block; margin: 0 auto; width: 100%;" />

Integrating Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) has yielded promising results in modeling spatiotemporal dynamics. However, CNNs are often constrained by limited local receptive fields, while RNNs struggle with high computational costs and difficulties in capturing long-term dependencies, both of which hinder model scalability and prediction accuracy. In this paper, we propose Mambaformer, a novel non-recurrent architecture that synergizes Visual Mamba2 and Transformer components. The core design philosophy assigns Mamba2 to capture temporal evolution efficiently, while leveraging the Transformer architecture to model complex spatial dependencies. This division of labor exploits the complementary strengths of both paradigms. Extensive experiments demonstrate that Mambaformer achieves state-of-the-art (SOTA) performance across three standard benchmarks. The significant accuracy improvements highlight the potential of Mambaformer as a robust baseline for real-world video prediction applications.

## Overview
- `arg_setting/` used to parse configuration files for spatio-temporal sequence-related tasks and generate a unified setting dictionary for the model training, validation, and testing processes.
- `configs/` YAML configuration file providing data.
- `datasets/` PyTorch Lightning data module for building datasets.
- `models/` an AttenRNN model that combines attention mechanism and recurrent neural network is implemented.
- `task/` predict and evaluate the prediction performance of different types of spatiotemporal data.
- `utils/` spati-temporal data processing and visualization.
- `test_w.py` testing models trained based on the PyTorch Lightning framework.
- `train_w.py` a general framework for training models.

## Requirements
- python >= 3.8
- pytorch: 1.10.0+
- pytorch-lightning: 1.6.0+
- numpy: 1.21.0+
- pandas: 1.3.0+
- einops: 0.4.0+
- torchvision: 0.11.0+
- matplotlib: 3.5.0+
- scikit-image: 0.19.0+
- tqdm: 4.62.0+
- torchmetrics: 0.7.0+
## Train
    python train.py /Mambaformer/configs/taxibj/mambaformer/seed=27.yaml
## Test
    python test.py path/to/setting=27.yaml
## Citation
