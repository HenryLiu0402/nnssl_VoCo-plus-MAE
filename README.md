# nnssl_VoCo-plus-MAE
NTUST_EE304 @ SSL3D Challenge

## Description
We aim to balance performance across segmentation and classification tasks.
To this end, we explored two SSL methods: Masked Autoencoder (MAE), which shows strong performance in segmentation tasks, and Volume Contrastive Learning (VoCo), which is more effective in classification settings.
We compared their individual performance, different training orders (MAE→VoCo vs. VoCo→MAE), and further investigated the impact of batch size, weight decay, and dataset filtering (full vs. modality-selected).
Through these experiments, we derived the current results reported below.

## Environment
GPU: NVIDIA RTX A6000、NVIDIA TITAN RTX
CUDA: 12.4
Python: 3.10.12
PyTorch: 2.6.0+cu124


## License
This project is based on [nnssl (MIC-DKFZ)](https://github.com/MIC-DKFZ/nnssl)
and distributed under the [Apache License 2.0](LICENSE).
