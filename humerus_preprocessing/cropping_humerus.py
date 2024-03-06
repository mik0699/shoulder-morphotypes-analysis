import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from loguru import logger
import numpy as np
import time
import pickle
import h5py
import sys

def find_max_dimensions_humerus(pos_array_hor_all_crop,pos_array_ver_all_crop,pos_array_prof_all_crop):
    """
    Questa funzione permette di trovare la dimensione massima della bounding box partendo dagli array che contengono
    le posizioni di cropping
    """
    
    max_hor_crop = 0
    max_ver_crop = 0
    max_prof_crop = 0

    for elem in pos_array_hor_all_crop:
        if len(elem) > 2:
            #print(f"{elem[1]-elem[0]}, {elem[3]-elem[2]}")
            if elem[1]-elem[0] > max_hor_crop:
                max_hor_crop = elem[1]-elem[0] 
            if elem[3]-elem[2] > max_hor_crop:
                max_hor_crop = elem[3]-elem[2] 
        elif len(elem) == 2:
            #print(elem[1]-elem[0])
            if elem[1]-elem[0] > max_hor_crop:
                max_hor_crop = elem[1]-elem[0] 

    for elem in pos_array_ver_all_crop:
        if len(elem) > 2:
            #print(f"{elem[1]-elem[0]}, {elem[3]-elem[2]}")
            if elem[1]-elem[0] > max_ver_crop:
                max_ver_crop = elem[1]-elem[0] 
            if elem[3]-elem[2] > max_ver_crop:
                max_ver_crop = elem[3]-elem[2] 
        elif len(elem) == 2:
            #print(elem[1]-elem[0])
            if elem[1]-elem[0] > max_ver_crop:
                max_ver_crop = elem[1]-elem[0] 

    for elem in pos_array_prof_all_crop:
        if len(elem) > 2:
            #print(f"{elem[1]-elem[0]}, {elem[3]-elem[2]}")
            if elem[1]-elem[0] > max_prof_crop:
                max_prof_crop = elem[1]-elem[0] 
            if elem[3]-elem[2] > max_prof_crop:
                max_prof_crop = elem[3]-elem[2] 
        elif len(elem) == 2:
            #print(elem[1]-elem[0])
            if elem[1]-elem[0] > max_prof_crop:
                max_prof_crop = elem[1]-elem[0] 
                
    # Aggiungo 2 per lasciare un po' di margine anche alla ct più grande
    return max_hor_crop+2,max_ver_crop+2,max_prof_crop+2

def cropping_humerus(hor_crop_pos,ver_crop_pos,prof_crop_pos,dim_bb):
    """
    Effettua il cropping di tutte le labelmap dell'omero nelle CT
    
    ARGS:
        hor_crop_pos (list of list of int): lista contenente, in ogni posizione, una lista contenente le posizioni in cui
            tagliare orizzontalmente ogni CT
        ver_crop_pos (list of list of int): lista contenente, in ogni posizione, una lista contenente le posizioni in cui
            tagliare verticalmente ogni CT
        prof_crop_pos (list of list of int): lista contenente, in ogni posizione, una lista contenente le posizioni in cui
            tagliare in profondità (slice) ogni CT
        dim_bb (list of int): lista di 3 valori contenente le dimensioni della Boungind Box massima nelle 3 direzioni
    RETURNS:
        cropped_cts (np.ndarray): numpy array contenente tutto il dataset di ct tagliate
    """
    
    left_humerus_label = 69
    right_humerus_label = 70
    all_cropped_cts = []
    for i in range(1,1251):
        pos = i-1
        if len(hor_crop_pos[pos]) != 0: 
            # Una volta lo faccio sempre
            shoulder_seg_data = np.asarray(nib.load(f"processing/{i}/shoulder_seg_res.nii").dataobj)
            shoulder_seg_humerus_data = np.where((shoulder_seg_data == left_humerus_label) | (shoulder_seg_data == right_humerus_label),1,0).astype(np.uint8)
            # if len(prof_crop_pos) >= 2: # Sia quelli con una spalla che con due
            cropped_ct = shoulder_seg_humerus_data[hor_crop_pos[pos][0]:hor_crop_pos[pos][1], ver_crop_pos[pos][0]:ver_crop_pos[pos][1], prof_crop_pos[pos][0]:prof_crop_pos[pos][1]]
            # print(f"Dimensione prima: {data.shape}")
            pad_rows_bef = int((dim_bb[0]-cropped_ct.shape[0])/2)
            pad_rows_aft = dim_bb[0] - pad_rows_bef - cropped_ct.shape[0]
            pad_cols_bef = int((dim_bb[1]-cropped_ct.shape[1])/2)
            pad_cols_aft = dim_bb[1] - pad_cols_bef - cropped_ct.shape[1]
            pad_prof_bef = int((dim_bb[2]-cropped_ct.shape[2])/2)
            pad_prof_aft = dim_bb[2] - pad_prof_bef - cropped_ct.shape[2]
            cropped_ct = np.pad(cropped_ct,pad_width=((pad_rows_bef,pad_rows_aft),(pad_cols_bef,pad_cols_aft),(pad_prof_bef,pad_prof_aft)))
            all_cropped_cts.append(cropped_ct)
            logger.info(f"Ritagliato omero {i}")
            if len(hor_crop_pos[pos]) == 4: # Solo quelli con 2 spalle, faccio un altro cropping
                cropped_ct = shoulder_seg_humerus_data[hor_crop_pos[pos][2]:hor_crop_pos[pos][3], ver_crop_pos[pos][2]:ver_crop_pos[pos][3], prof_crop_pos[pos][2]:prof_crop_pos[pos][3]]
                # print(f"Dimensione prima: {data.shape}")
                pad_rows_bef = int((dim_bb[0]-cropped_ct.shape[0])/2)
                pad_rows_aft = dim_bb[0] - pad_rows_bef - cropped_ct.shape[0]
                pad_cols_bef = int((dim_bb[1]-cropped_ct.shape[1])/2)
                pad_cols_aft = dim_bb[1] - pad_cols_bef - cropped_ct.shape[1]
                pad_prof_bef = int((dim_bb[2]-cropped_ct.shape[2])/2)
                pad_prof_aft = dim_bb[2] - pad_prof_bef - cropped_ct.shape[2]
                cropped_ct = np.pad(cropped_ct,pad_width=((pad_rows_bef,pad_rows_aft),(pad_cols_bef,pad_cols_aft),(pad_prof_bef,pad_prof_aft)))
                all_cropped_cts.append(cropped_ct)
                logger.info(f"Ritagliato omero {i} seconda volta")
    all_cropped_cts = np.asarray(all_cropped_cts)
    return all_cropped_cts


if __name__ == "__main__":
    st = time.time()
    logger.remove()
    logger.add("humerus_preprocessing/cropping_humerus_res_log.log", format="{time:DD-MM HH:mm:ss} - {message}", level="INFO")
    logger.add(sys.stdout, format="{message}", level="SUCCESS")

    # Leggo gli array
    with open(f"humerus_preprocessing/arrays/pos_array_hor_all.pkl","rb") as f_hor,open(f"humerus_preprocessing/arrays/pos_array_ver_all.pkl","rb") as f_ver,open(f"humerus_preprocessing/arrays/pos_array_prof_all.pkl","rb") as f_prof:
            pos_array_hor_all_crop = pickle.load(f_hor)
            pos_array_ver_all_crop = pickle.load(f_ver)
            pos_array_prof_all_crop = pickle.load(f_prof)

    # Trovo il massimo della bounding box      
    dim_bb = find_max_dimensions_humerus(pos_array_hor_all_crop,pos_array_ver_all_crop,pos_array_prof_all_crop)
    # Effettuo il cropping        
    cropped_cts = cropping_humerus(pos_array_hor_all_crop,pos_array_ver_all_crop,pos_array_prof_all_crop,dim_bb)
    logger.success(f"Tempo totale cropping: {(time.time()-st):.2f} sec")

    st = time.time()
    # Salvo il dataset
    with h5py.File("dataset_humerus_labelmap_res.hdf5","w") as f:
        data_set = f.create_dataset("mydataset",data=cropped_cts)

    logger.success(f"Tempo totale salvataggio: {(time.time()-st):.2f} sec")