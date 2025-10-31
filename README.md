# KVGAE: revealing heterogeneity in spatial transcriptomics through capturing structured long range interactions

Introduction
This paper proposes an unsupervised learning method—KVGAE. Firstly, we construct a multimodal weight matrix by integrating image features, gene expression, and spatial location information, and perform weighted aggregation between each spatial spot and its local neighborhood to obtain a more stable expression representation. Secondly, we combine a multi-head attention mechanism with a Fourier basis-driven Kolmogorov-Arnold Network to jointly model long-range dependencies and periodic patterns, while enhancing the interpretability of the model. Finally, across eight datasets from four sequencing platforms, KVGAE demonstrates superior accuracy in spatial domain identification.

## Installation
```bash
python == 3.9
torch == 1.13.0
scanpy == 1.9.2
anndata == 0.8.0
numpy == 1.22.3
```

## Dataset
```bash
(1) Human DLPFCs within the spatialLIBD at http://research.libd.org/spatialLIBD/
(2) Human breast cancer dataset at https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Breast_Cancer_Block_A_Section_1/V1_Breast_Cancer_Block_A_Section_1_web_summary.html
(3) Mouse embryo E9.5 data captured using stereo-seq at https://db.cngb.org/stomics/mosta/
(4) Mouse brain tissue datasets from the 10× Genomics database, including sagittal anterior, sagittal posterior, and coronal sections of adult mouse brain, are available at https://www.10xgenomics.com/
(5) Mouse somatosensory cortex data obtained using the osmFISH platform at http://linnarssonlab.org/osmFISH/
(6) Mouse visual cortex data generated using the STARmap platform is available at https://singlecell.broadinstitute.org/single_cell/study/SCP815
```

## Run
```bash
from RUN import RunAnalysis
import scanpy as sc



path = "/root/lijie/dataset/human breast/human breast datast"  # Replace with the actual path
count_file = "V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5"
image_path = path + "/spatial/tissue_hires_image.png"  


save_path = "../result"
n_domains = 7


adata = read_10X_Visium(
    path=path,
    count_file=count_file,
    load_images=True,
    quality="hires", 
    image_path=image_path)


runner = RunAnalysis(save_path=save_path, pre_epochs=1000, epochs=600, use_gpu=True)

Crop histological section
adata = runner.crop_image(adata, data_name="human_breast", crop_size=50, target_size=224)

Feature extraction from images
adata = runner.extract_image_features(adata, cnn_type="MaxVit", pca_components=50)

Data augmentation
adata = runner.augment_adata(adata)

PCA dimensionality reduction
data = runner.data_process(adata, pca_n_comps=200)

Graph construction
graph_dict = runner.build_graph(adata, k=8, rad_cutoff=150)

Model construction
model = runner.build_model(input_dim=data.shape[1])

Model training
trainer = runner.train_model(data, graph_dict, model)
trainer.fit()

emb, _ = trainer.process()
adata.obsm["emb"] = emb

Clustering
adata = runner.get_cluster_data(adata, n_domains=n_domains, priori=True)

sc.pl.spatial(adata, color='refine spatial domain',  spot_size=150)
```
