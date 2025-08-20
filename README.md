# nnssl_VoCo-plus-MAE
NTUST_EE304 @SSL3D Challenge
ResEnc-L network architecture

## Description
We aim to balance performance across segmentation and classification tasks.
To this end, we explored two SSL methods: Masked Autoencoder (MAE), which shows strong performance in segmentation tasks, and Volume Contrastive Learning (VoCo), which is more effective in classification settings.
We compared their individual performance, different pre-training orders (MAE→VoCo vs. VoCo→MAE), and further investigated the impact of batch size, weight decay, and dataset filtering (full vs. modality-selected).
Through these experiments, we derived the current results reported below.

## Environment
GPU: NVIDIA RTX A6000、NVIDIA TITAN RTX  
CUDA: 12.4  
Python: 3.10.12  
PyTorch: 2.6.0+cu124  

## Pretraining Straetegy
Our final pretraining strategy consists of a two-stage pipeline:  
### Stage 1–VoCo Pretraining (500 epochs, all modalities)
We first apply Volume Contrastive Learning (VoCo) on the full OpenMind dataset, including all available MRI modalities.
This allows the model to learn from a broad range of contrasts, which maximizes representation diversity and stability, and improves generalization, especially for classification tasks, through contrastive learning.

### Stage 2–MAE Pretraining (500 epochs, filtered modalities)

We then continue with Masked Autoencoder (MAE) pretraining on a filtered subset of modalities **("T1w", "inplainT1", "MP2RAGE", "FLAIR", "T2w", "inplainT2", "ADC", "DWI")**, which are most relevant to downstream segmentation and classification tasks.
MAE promotes fine-grained feature learning by reconstructing masked inputs, which better supports the requirements of segmentation.  
By focusing on key modalities, we reduce noise from less common or unrelated sequences, allowing the model to concentrate its representation capacity on the most task-relevant contrasts.

## Hyperparameter Settings
For VoCo pre-training stage:  
batch_size = 8  
initial_lr = 1e-2  
weight_decay = 3e-5  
num_epochs = 500  

For MAE pre-training stage:  
batch_size = 4  
patch_size = (160, 160, 160)  
mask ratio: 0.75  
num_epochs = 500  
filtered modalities: "T1w", "inplainT1", "MP2RAGE", "FLAIR", "T2w", "inplainT2", "ADC", "DWI"

## License
This project is based on [nnssl (MIC-DKFZ)](https://github.com/MIC-DKFZ/nnssl)
and distributed under the [Apache License 2.0](LICENSE).
