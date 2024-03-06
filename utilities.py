import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

def plot_slices(data,nrows,ncols,starting_slice=0,title="",titlesize=60,axes=False):
    """
    La seguente funzione permette di stampare un certo numero di slice di una CT
    
    ARGS:
        data (np.ndarray): array contenente i dati della CT da stampare
        nrows (int): numero di righe
        ncols (int): numero di colonne
        starting_slice (int) [opzionale]: numero di slice da cui partire per la stampe
    """
    nslices = data.shape[2]
    data = data[:,:,starting_slice:]
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize = (40,int(40/ncols)*nrows),
        constrained_layout=True # Per mettere tutto vicino
    )
    if title:
        fig.suptitle(title,size=titlesize)
    if nrows == 1:
        for j in range(ncols):
            index = j
            if index < nslices - starting_slice:
                ax[j].imshow(data[:,:,index],cmap="gray")
            ax[j].axis("on") if axes else ax[j].axis("off")
            ax[j].tick_params(labelsize=25)
    else:       
        for i in range(nrows):
            for j in range(ncols):
                index = i*ncols+j
                if index < nslices - starting_slice:
                    ax[i,j].imshow(data[:,:,index],cmap="gray")
                ax[i,j].axis("on") if axes else ax[i,j].axis("off")
                ax[i,j].tick_params(labelsize=25)
                
def plot_all_slices_notzero(data,dim=2,title="",axes=False):
    """
    Stampa tutte le slice di una CT che contengono informazioni
    ARGS
        dim (int) [default=2]: la dimensione lungo cui stampare le slice. Valore in [0,1,2]
    """
    if dim not in [0,1,2]:
        print("La dimensione deve essere un valore in [0,1,2]")
        return
    notzero_mask = np.max(data,axis=(0,1) if dim==2 else (0,2) if dim == 1 else (1,2))
    notzero_slices = (notzero_mask!=0).sum()
    print(f"Slice diverse da zero: {notzero_slices}")
    data = data[:,:,notzero_mask!=0] if dim==2 else data[:,notzero_mask!=0,:] if dim==1 else data[notzero_mask!=0,:,:]
    ncols = 10
    nrows = int(np.ceil(notzero_slices/ncols))
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize = (40,int(40/ncols)*nrows),
        constrained_layout=True # Per mettere tutto vicino
    )
    if title:
        fig.suptitle(title,size=60)
    for i in range(nrows):
        for j in range(ncols):
            if nrows != 1:
                index = i*ncols+j
                if index < notzero_slices:
                    ax[i,j].imshow(data[:,:,index] if dim==2 else data[:,index,:] if dim==1 else data[index,:,:],cmap="gray")
                ax[i,j].axis("on") if axes else ax[i,j].axis("off")
                ax[i,j].tick_params(labelsize=25)
            else:
                index = j
                if index < notzero_slices:
                    ax[j].imshow(data[:,:,index] if dim==2 else data[:,index,:] if dim==1 else data[index,:,:],cmap="gray")
                ax[j].axis("on") if axes else ax[j].axis("off")
                ax[j].tick_params(labelsize=25)

                
def dice_coef(y_true, y_pred, smooth=100): # Smooth per evitare divisione per zero 
    """
    Calcola il dice coefficient facendo il flatten
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_coef_2(y_true, y_pred, smooth=100):
    """
    Calcola il dice coefficient senza flattern, utilizzando axis. Attenzione alle dimensioni degli input
    """
    # Stessa cosa ma senza i flatten, risultato uguale, solo che qui devo stare attento alle dim degli input (con squeeze e expand)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3,4])
    union = K.sum(y_true, axis=[1,2,3,4]) + K.sum(y_pred, axis=[1,2,3,4])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)