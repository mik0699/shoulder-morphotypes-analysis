import io
import base64
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
#from loguru import logger
import numpy as np
import time
import pickle
import h5py
#from skimage.transform import downscale_local_mean,resize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score
import seaborn as sns
from sklearn.cluster import KMeans,DBSCAN
from sklearn.neighbors import NearestNeighbors
import umap
import scipy
import time
import pandas as pd

from dash import Dash, dcc, html, Input, Output, no_update, callback, jupyter_dash
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html
import dash_vtk
from dash_vtk.utils import to_volume_state,to_mesh_state
import vtk
import dash_bootstrap_components as dbc

from PIL import Image

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

import itk
from itkwidgets import view
import SimpleITK as sitk

from utilities import plot_all_slices_notzero
    
# Ora non usata
def np_image_to_base64(humerus_dataset,slice_x,slice_y,slice_z):
    im_x = Image.fromarray(np.pad(humerus_dataset[slice_x,:,:],((22,21),(0,0)))*255)
    im_y = Image.fromarray(humerus_dataset[:,slice_y,:]*255)
    im_z = Image.fromarray(np.pad(humerus_dataset[:,:,slice_z],((0,0),(152,151)))*255)
    
    img_conc = Image.new("L",(506,246*3))
    img_conc.paste(im_x,(0,0))
    img_conc.paste(im_y,(0,246))
    img_conc.paste(im_z,(0,246*2))
    
    buffer = io.BytesIO()
    img_conc.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


if __name__=="__main__":
    with open("processing/humerus_res_flip_features_250ep.pkl","rb") as f:
        features = pickle.load(f)
    
    with open("hum_pos_array_hor_all.pkl","rb") as f1:
        pos_array_hor_all = pickle.load(f1)
        
    dataset_el_to_ct = [] # In corrispondenza dell'indice l'elemento conterrà la CT corrispondente
    for i,el in enumerate(pos_array_hor_all,start=1):
        if len(el) != 0:
            dataset_el_to_ct.append(i)
        if len(el) > 2:
            dataset_el_to_ct.append(i)
            
    number_clusters = 10
    kmeans = KMeans(n_clusters=number_clusters,n_init="auto",random_state=42)
    labels = kmeans.fit_predict(features)
    
    # TSNE
    st = time.time()
    features_tsne = TSNE(random_state=42).fit_transform(features) # Per la visualizzazione in 2D
    print(f"Tempo dimensionality reduction TSNE: {time.time()-st:.2f} sec")
    print(features_tsne.shape)
    
    # UMAP
    st = time.time()
    features_umap = umap.UMAP(n_neighbors=30,random_state=42).fit_transform(features)
    print(f"Tempo dimensionality reduction UMAP: {time.time()-st:.2f} sec")
    print(features_umap.shape)
    
    with open("processing/humerus_volumes.pkl","rb") as f:
        rel_vols = pickle.load(f)

    use_tsne = False # True uso TSNE, False uso UMAP
    # Meglio creare un dataFrame per i dati
    data = {
        "feature1": features_umap[:,0] if not use_tsne else features_tsne[:,0],
        "feature2": features_umap[:,1] if not use_tsne else features_tsne[:,1],
        "cluster": labels.astype(str),
        "volume": rel_vols
    }
    df = pd.DataFrame(data)
    # DASH app
    app = Dash(__name__)

    app.layout = html.Div(
        className="container",
        children=[
            html.Div([
                html.P("Modalità di visualizzazione:"),
                dcc.RadioItems(
                    id='graph-info', 
                    value='Cluster', 
                    options=['Cluster', 'Volumi'],
                    inline=True,
                )
            ],
            style={"margin-left":"5em"}),
            # dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
            dcc.Graph(id="graph-5", clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-5", direction='left'),
        ],
    )

    @app.callback(
        Output("graph-tooltip-5", "show"),
        Output("graph-tooltip-5", "bbox"),
        Output("graph-tooltip-5", "children"),
        Input("graph-5", "hoverData")
    )
    def display_hover(hoverData):
        # print(hoverData)
        if hoverData is None:
            return False, no_update, no_update

        hover_data = hoverData["points"][0]
        # print(hover_data)
        bbox = hover_data["bbox"]
        # num = hover_data["pointNumber"] # Con cluster non funziona
        x_coord = hover_data['x']
        num = df[df.feature1==x_coord].index[0] # Occhio che se ci sono due punti con la stessa x non funziona, qui però non avviene

        #im_matrix = humerus_dataset[num] # Qui va la CT, num è il numero del pallino su cui si passa il puntatore
        #im_url = np_image_to_base64(im_matrix,120,120,250)
        children = [
            html.Div([
                #html.Img(src=im_url,style={"width": "120px", 'display': 'block', 'margin': '0 auto'}),
                html.P("CT " + str(dataset_el_to_ct[num]), style={'font-weight': 'bold'}),
                html.P(f"Volume: {rel_vols[num]:.3f}")
            ])
        ]

        return True, bbox, children

    @app.callback(
        Output("graph-5", "figure"),
        Input("graph-info", "value")
    )
    def change_mode(graph_info):
        # Figura dinamica
        fig = px.scatter(
                df,
                x="feature1",
                y="feature2",
                color="volume" if graph_info=="Volumi" else "cluster", # Trasformo in stringhe per avere discreto
            )

        fig.update_traces(
            hoverinfo="none",
            hovertemplate=None,
        )

        fig.update_layout(
            autosize=False,
            width=1450,
            height=800
        )

        return fig

    app.title = "Humerus TSNE" if use_tsne else "Humerus UMAP"
    app.run(jupyter_mode="external",host="0.0.0.0",port="8050")
