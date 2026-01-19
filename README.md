# AROMMA: Unifying Olfactory Embeddings for Single Molecules and Mixtures
## Introduction
This is the pytorch implementation of our ICASSP 2026 paper "[AROMMA: Unifying Olfactory Embeddings for Single Molecules and Mixtures]()". **AROMMA (Aggregated Representations of Olfaction via Molecule and Mixture Alignment)** is a novel framework that learns a **unified embedding space** for both single molecules and two-molecule mixtures by **leveraging chemical foundation model ([SPMM](https://doi.org/10.1038/s41467-024-46440-3))**. 

To address the **label sparsity** in the mixture dataset (BP), AROMMA employs a training strategy that combines:
- **Knowledge distillation** from a molecule-level teacher model ([POM](https://doi.org/10.1088/2632-2153/adfffc)), and  
- **Class-distribution-aware pseudo-labeling**.

## Setup
Create and activate the conda environment:
```
conda env create -f environment.yml
conda activate aromma_env
```

## Download trained checkpoints 
The trained model checkpoints are available on Hugging Face:
|Data|Model|Checkpoints|
|:--:|:--:|:--:|
|`data/mixture`|AROMMA|[aromma_best_fold.pt](https://huggingface.co/riverallzero/aromma/blob/main/aromma_best_fold.pt)|
|`data/mixture_p78`|AROMMA-P78|[aromma_p78_best_fold.pt](https://huggingface.co/riverallzero/aromma/blob/main/aromma_p78_best_fold.pt)|
|`data/mixture_p152`|AROMMA-P152|[aromma_p152_best_fold.pt](https://huggingface.co/riverallzero/aromma/blob/main/aromma_p152_best_fold.pt)

## Training
Before training, download the pre-trained checkpoints following `models/pom/README.md` and `models/spmm/README.md`
The directory structure should be:
```
models
├── pom
│   ├── gnn_embedder.pt
│   └── nn_predictor.pt
└── spmm
    ├── checkpoint_SPMM.ckpt
    ├── config_bert.json
    └── vocab_bpe_300.txt
```

Training Procedure:
1. `python train.py --phase aromma`
2. `pseudo_labeling.ipynb`
3. `python train.py --phase aromma_p78` or `python train.py --phase aromma_p152`

## Citation

```
```
