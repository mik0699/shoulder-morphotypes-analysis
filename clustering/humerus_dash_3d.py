from dash import Dash, dcc, html, Input, Output, no_update, callback, jupyter_dash
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html
import dash_vtk
from dash_vtk.utils import to_volume_state,to_mesh_state
import vtk
import dash_bootstrap_components as dbc
import pickle
import os

def create_col(title,image,wid_cols=4,heig=300):
    """
    Crea un'immagine della griglia con titolo e renering
    """
    column = dbc.Col(
                html.Div([
                    html.H4(children=title),
                    html.Div(
                    style={"width": "100%", "height": f"{heig}px"},
                    children=[image]
                    ),
                ]),
                width=wid_cols
            )
    return column

def create_row(cluster,use_flipped=True,show_flip=True):
    """
    Ritorna le tre immagini più vicine al centroide del cluster passato come parametro e i rispettivi numeri di CT 
    """
    reader = vtk.vtkDataSetReader() # Devo usare questo reader
    if use_flipped:
        ct_0 = indexes_ct_flip[cluster][0]
        ct_1 = indexes_ct_flip[cluster][1]
        ct_2 = indexes_ct_flip[cluster][2]
    else:
        ct_0 = indexes_ct[cluster][0]
        ct_1 = indexes_ct[cluster][1]
        ct_2 = indexes_ct[cluster][2]
    
    if use_flipped:
        if show_flip and os.path.exists(f"vtk_images/humerus_{ct_0}_segm_flip_mirrored.vtk"):
            reader.SetFileName(f"vtk_images/humerus_{ct_0}_segm_flip_mirrored.vtk")
        else:
            reader.SetFileName(f"vtk_images/humerus_{ct_0}_segm_flip.vtk")
    else:
        reader.SetFileName(f"vtk_images/humerus_{ct_0}_segm.vtk")
    reader.Update()
    volume_state_0 = to_volume_state(reader.GetOutput())

    humerus_0 = dash_vtk.View([
        dash_vtk.VolumeRepresentation(
            children=[
                dash_vtk.VolumeController(),
                dash_vtk.Volume(state=volume_state_0),
            ],
        ),
    ])
    
    if use_flipped:
        if show_flip and os.path.exists(f"vtk_images/humerus_{ct_1}_segm_flip_mirrored.vtk"):
            reader.SetFileName(f"vtk_images/humerus_{ct_1}_segm_flip_mirrored.vtk")
        else:
            reader.SetFileName(f"vtk_images/humerus_{ct_1}_segm_flip.vtk")
    else:
        reader.SetFileName(f"vtk_images/humerus_{ct_1}_segm.vtk")
    reader.Update()
    volume_state_1 = to_volume_state(reader.GetOutput())

    humerus_1 = dash_vtk.View([
        dash_vtk.VolumeRepresentation([
            dash_vtk.VolumeController(),
            dash_vtk.Volume(state=volume_state_1),
        ]),
    ])
    
    if use_flipped:
        if show_flip and os.path.exists(f"vtk_images/humerus_{ct_2}_segm_flip_mirrored.vtk"):
            reader.SetFileName(f"vtk_images/humerus_{ct_2}_segm_flip_mirrored.vtk")
        else:
            reader.SetFileName(f"vtk_images/humerus_{ct_2}_segm_flip.vtk")
    else:
        reader.SetFileName(f"vtk_images/humerus_{ct_2}_segm.vtk")
    reader.Update()
    volume_state_2 = to_volume_state(reader.GetOutput())

    humerus_2 = dash_vtk.View([
        dash_vtk.VolumeRepresentation([
            dash_vtk.VolumeController(),
            dash_vtk.Volume(state=volume_state_2),
        ]),
    ])
    
    return humerus_0,humerus_1,humerus_2,ct_0,ct_1,ct_2

if __name__=="__main__":
    # Con 2 callback diversi riesco a separare i due caricamenti e aggiornare solo la riga che cambia
    with open("processing/humerus_indexes_ct.pkl","rb") as f:
        indexes_ct = pickle.load(f)

    with open("processing/humerus_indexes_ct_flip.pkl","rb") as f:
        indexes_ct_flip = pickle.load(f)
        
    # Con 2 callback diversi riesco a separare i due caricamenti e aggiornare solo la riga che cambia
    use_flip = True
    show_flip=True

    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

    dropdown_style = {
        "width":"50%",
        "margin-top":"1em",
        "margin-bottom":"1em",
    }
    dropdown_list = [f"Cluster {x}" for x in range(len(indexes_ct_flip))] if use_flip else [f"Cluster {x}" for x in range(len(indexes_ct))]

    app.layout = dbc.Container([
        dcc.Dropdown(dropdown_list,"Cluster 0",id="dropdown_1",style=dropdown_style,clearable=False),
        dbc.Row(id="row_1"),
        dcc.Dropdown(dropdown_list,"Cluster 1",id="dropdown_2",style=dropdown_style,clearable=False),
        dbc.Row(id="row_2",style={"margin-bottom":"5em"}),
    ])

    @app.callback(
        Output("row_1","children"),
        Input("dropdown_1","value"),
    )
    def update_row_1(drop_1):  
        cluster_1 = int(drop_1.split()[1])

        humerus_1_1,humerus_1_2,humerus_1_3,ct_1_1,ct_1_2,ct_1_3 = create_row(cluster_1,use_flipped=use_flip,show_flip=show_flip)
        row_1 = [
            create_col(f"Cluster {cluster_1} Omero {ct_1_1}",humerus_1_1),
            create_col(f"Cluster {cluster_1} Omero {ct_1_2}",humerus_1_2),
            create_col(f"Cluster {cluster_1} Omero {ct_1_3}",humerus_1_3),
        ]

        return row_1

    @app.callback(
        Output("row_2","children"),
        Input("dropdown_2","value"),
    )
    def update_row_2(drop_2):  
        cluster_2 = int(drop_2.split()[1])

        humerus_2_1,humerus_2_2,humerus_2_3,ct_2_1,ct_2_2,ct_2_3 = create_row(cluster_2,use_flipped=use_flip,show_flip=show_flip)
        row_2 = [
            create_col(f"Cluster {cluster_2} Omero {ct_2_1}",humerus_2_1),
            create_col(f"Cluster {cluster_2} Omero {ct_2_2}",humerus_2_2),
            create_col(f"Cluster {cluster_2} Omero {ct_2_3}",humerus_2_3),
        ]

        return row_2

    app.title = "Humerus 3D Rendering"
    app.run(jupyter_mode="external",host="0.0.0.0",port="2323")