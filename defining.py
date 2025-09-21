import os
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from typing import List




#Read 10X Visium data
def read_10X_Visium(
    path,
    genome=None,
    count_file='filtered_feature_bc_matrix.h5',
    library_id=None,
    load_images=True,
    quality='hires',
    image_path=None
):
    adata = sc.read_visium(
        path,
        genome=genome,
        count_file=count_file,
        library_id=library_id,
        load_images=load_images,
    )
    adata.obsm["spatial"] = np.array(adata.obsm["spatial"], dtype=float)
    adata.var_names_make_unique()

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale

    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality

    return adata




#The optimal resolution parameter for the expected number of clusters was calculated
def _priori_cluster(
        
        adata,
        n_domains=7,
    ):
        
        for res in sorted(list(np.arange(0.1, 2.5, 0.01)), reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            if count_unique_leiden == n_domains:
                break
        return res


#Try different resolution parameters on the given data and choose the best one for clustering to improve the quality of clustering.
def _optimize_cluster(
        
        adata,
        resolution: list = list(np.arange(0.1, 2.5, 0.01)),
    ):
        scores = []
        for r in resolution:
            #莱顿聚类
            sc.tl.leiden(adata, resolution=r)
            s = calinski_harabasz_score(adata.X, adata.obs["leiden"])
            scores.append(s)
        cl_opt_df = pd.DataFrame({"resolution": resolution, "score": scores})
        best_idx = np.argmax(cl_opt_df["score"])
        res = cl_opt_df.iloc[best_idx, 0]
        print("Best resolution: ", res)
        return res

#Neighborhood smoothing was performed on the predicted labels for each spatial point
def refine(
    sample_id: List,
    pred: List,
    dis: np.ndarray,
    shape: str = "hexagon"
) -> List:
    refined_pred = []
    pred_df = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)

    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        raise ValueError("Shape not recognized. Use 'hexagon' for Visium data or 'square' for ST data.")

    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values()
        nbs = dis_tmp[0:num_nbs + 1]
        nbs_pred = pred_df.loc[nbs.index, "pred"]
        self_pred = pred_df.loc[index, "pred"]
        v_c = nbs_pred.value_counts()

        if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) > num_nbs / 2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)

    return refined_pred


