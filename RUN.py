import os, time, psutil
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import distance 
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial import distance
from defining import read_10X_Visium
from image_featurer import  ImageFeature, image_crop
from graph import graph, combine_graph_dict
from model import KVGAE_model
from train import Train
from augment import augment_adata
from defining import _optimize_cluster
from defining import _priori_cluster
from defining import refine



class RunAnalysis:
    def __init__(self, save_path="./", pre_epochs=300, epochs=400, use_gpu=True):
        self.save_path = save_path
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.use_gpu = use_gpu
        self.model = None
        self.trainer = None

    # Crop histological section
    def crop_image(
        self, adata, data_name: str,
        library_id=None, crop_size: int = 50, target_size: int = 224, verbose: bool = False
    ):
        out_dir = Path(self.save_path) / "Image_crop" / data_name
        out_dir.mkdir(parents=True, exist_ok=True)
        adata = image_crop(
            adata=adata,
            save_path=out_dir,
            library_id=library_id,
            crop_size=crop_size,
            target_size=target_size,
            verbose=verbose,
        )
        return adata

    # Feature extraction from images
    def extract_image_features(
        self, adata,
        cnn_type: str = "MaxVit", pca_components: int = 50,
        verbose: bool = False, seeds: int = 88, ensure_cropped: bool = True
    ):
        if ensure_cropped and "slices_path" not in adata.obs:
            raise ValueError("未发现 adata.obs['slices_path']，请先执行 crop_image。")
        extractor = ImageFeature(
            adata=adata,
            pca_components=pca_components,
            cnn_type=cnn_type,
            verbose=verbose,
            seeds=seeds,
        )
        return extractor.extract_image_feat()

    # Data augmentation
    def augment_adata(
        self, adata,
        md_dist_type="cosine", gb_dist_type="correlation", n_components=50,
        use_morphological=True, use_data="raw", neighbour_k=6, adjacent_weight=0.2,
        spatial_k=30, spatial_type="KDTree"
    ):
        return augment_adata(
            adata=adata,
            md_dist_type=md_dist_type,
            gb_dist_type=gb_dist_type,
            n_components=n_components,
            use_morphological=use_morphological,
            use_data=use_data,
            neighbour_k=neighbour_k,
            adjacent_weight=adjacent_weight,
            spatial_k=spatial_k,
            spatial_type=spatial_type,
        )

    # PCA dimensionality reduction
    def data_process(self, adata, pca_n_comps=200):
        adata.raw = adata
        adata.X = adata.obsm["augment_gene_data"].astype(np.float64)
        data = sc.pp.normalize_total(adata, target_sum=1, inplace=False)['X']
        data = sc.pp.log1p(data)   
        data = sc.pp.scale(data)   
        data = sc.pp.pca(data, n_comps=pca_n_comps)
        return data

    # Graph construction
    def build_graph(self, adata, k=8, rad_cutoff=150, distType="KDTree"):
        g = graph(adata.obsm["spatial"], k=k, rad_cutoff=rad_cutoff, distType=distType)
        graph_dict = g.main()
        return graph_dict

    # Model construction
    def build_model(
        self, input_dim,
        Conv_type="GCNConv",
        linear_encoder_hidden=[64, 40],
        linear_decoder_hidden=[64],
        conv_hidden=[64, 16],
        p_drop=0.01,
        dec_cluster_n=20,
    ):
        self.model = KVGAE_model(
            input_dim=input_dim,
            Conv_type=Conv_type,
            linear_encoder_hidden=linear_encoder_hidden,
            linear_decoder_hidden=linear_decoder_hidden,
            conv_hidden=conv_hidden,
            p_drop=p_drop,
            dec_cluster_n=dec_cluster_n,
        )
        return self.model  

    def train_model(
        self, data, graph_dict, model=None,
        pre_epochs=1000, epochs=600,
        kl_weight=1, mse_weight=1, bce_kld_weight=1, domain_weight=1,
        use_gpu=True,
    ):
        if model is None:
            if self.model is None:
                raise ValueError("请先 build_model() 或传入 model。")
            model = self.model
        self.trainer = Train(
            processed_data=data,
            graph_dict=graph_dict,
            model=model,
            pre_epochs=pre_epochs,
            epochs=epochs,
            kl_weight=kl_weight,
            mse_weight=mse_weight,
            bce_kld_weight=bce_kld_weight,
            domain_weight=domain_weight,
            use_gpu=use_gpu,
        )
        return self.trainer  
    # 
    def cluster_with_refine(self, adata, n_domains, priori=True):
        sc.pp.neighbors(adata, use_rep='emb')
        if priori:
            res = _priori_cluster(adata, n_domains=n_domains)
        else:
            res = _optimize_cluster(adata)
        sc.tl.leiden(adata, key_added="spatial domain", resolution=res)

        adj_2d = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')
        refined_pred = refine(
            sample_id=adata.obs.index.tolist(),
            pred=adata.obs["spatial domain"].tolist(),
            dis=adj_2d,
            shape="hexagon"
        )
        adata.obs["refine spatial domain"] = refined_pred
        return adata
