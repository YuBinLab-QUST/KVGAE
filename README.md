# KVGAE: A variational graph autoencoder integratiing attention and fourier-kan structures reveals heterogeneity in spatial transccriptomics

##Introduction
This paper proposes an unsupervised learning method—KVGAE. Firstly, we construct a multimodal weight matrix by integrating image features, gene expression, and spatial location information, and perform weighted aggregation between each spatial spot and its local neighborhood to obtain a more stable expression representation. Secondly, we combine a multi-head attention mechanism with a Fourier basis-driven Kolmogorov-Arnold Network to jointly model long-range dependencies and periodic patterns, while enhancing the interpretability of the model. Finally, across eight datasets from four sequencing platforms, KVGAE demonstrates superior accuracy in spatial domain identification.

##Installation
```bash
python == 3.9
torch == 1.13.0
scanpy == 1.9.2
anndata == 0.8.0
numpy == 1.22.3
```
