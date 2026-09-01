# ScaMLLM

This repository contains the implementation and supplementary resources for our paper **“ScaMLLM: A Multimodal Large Language Model for Scaffold Constrained and Property Controllable Molecular Generation.”**

## Installation

Please create and configure the Python virtual environment according to the dependencies specified in `requirements.txt`.

```bash
pip install -r requirements.txt
```

We recommend using a dedicated virtual environment to avoid potential dependency conflicts.

## Pre-trained Model Weights

Before running the code, please download all required pre-trained model weights and place them in the corresponding directories under the `models/` folder.

The expected directory structure is:

```text
models/
├── ...
├── ...
└── ...
```

Please make sure that the downloaded model weights are placed in the correct locations before running the training or inference scripts.

## Dataset Availability

The `datasets/scaffold_images` directory contains the scaffold image data used in our experiments.

Due to the current data availability and release conditions, the **`scaffold_images` dataset will be made available after the paper is officially accepted for publication**.

The remaining datasets and resources required to reproduce our experiments will be provided in accordance with their respective licenses and availability conditions.

## Repository Updates

This repository is currently under active development. Additional code, datasets, configuration files, documentation, and other resources will be gradually updated and released in this repository.

Please check this repository regularly for the latest updates.

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{yang2026scamllm, title = {ScaMLLM: A Multimodal Large Language Model for Scaffold Constrained and Property Controllable Molecular Generation},
author = {Yang, Shuting and Yu, Zihao and Cheng, Debo and Huang, Yu and Feng, Zaiwen and Li, Chen},
year = {2026} }
```

More details and complete reproduction instructions will be provided as the repository is updated.
