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
import utilities
from itkwidgets import view

# Troviamo la bounding box più grande tra tutte le ct (all'inizio solo le prime 100)
def find_largest_box_humerus():
    excluded_cts = [167,233,339,342,343,685,752,1003,1152] # CT che danno problemi
    left_humerus_label = 69
    right_humerus_label = 70
    left_scapula_label = 71
    right_scapula_label = 72
    pos_array_hor_all = []
    pos_array_ver_all = []
    pos_array_prof_all = []
    max_width = 0
    max_height = 0
    max_slices = 0
    limit = 50 
    limit_slices = 20 # Le CT con meno di questo numero di slice sono escluse perchè solitamente sono errori
    
    for i in range(1,1251): 
        pos_array_prof_ct = []
        pos_array_hor_ct = []
        pos_array_ver_ct = []
        if os.path.exists(f"processing/{i}/shoulder_seg_res.nii"): 
            # Saranno poi da leggere preventivamente e mettere in un Dataset, oppure si mettono quelli già processati
            shoulder_seg_data = np.asarray(nib.load(f"processing/{i}/shoulder_seg_res.nii").dataobj)
   
            if i not in excluded_cts:
                unique_labels = np.unique(shoulder_seg_data) 
                if (left_humerus_label in unique_labels) and (left_scapula_label in unique_labels): # Spalla sinistra 
                    shoulder_seg_humerus_data = np.where(shoulder_seg_data == left_humerus_label,1,0).astype(np.uint8)
                    shoulder_seg_scapula_data = np.where(shoulder_seg_data == left_scapula_label,1,0).astype(np.uint8)

                    max_prof_hum = np.max(shoulder_seg_humerus_data,axis=(0,1)) # Asse z
                    pos_array_prof_hum = np.diff(max_prof_hum,prepend=0).nonzero()[0] 

                    max_prof_scap = np.max(shoulder_seg_scapula_data,axis=(0,1)) # Asse z
                    pos_array_prof_scap = np.diff(max_prof_scap,prepend=0).nonzero()[0]

                    if pos_array_prof_scap.size != 0 and pos_array_prof_hum[-1]-pos_array_prof_scap[0] >= limit_slices: 
                        logger.info(f"Processo ct {i}: spalla sinistra")

                        # molto bassa
                        pos_array_prof_ct.append(pos_array_prof_scap[0])
                        pos_array_prof_ct.append(pos_array_prof_hum[-1])                         
                        # Per l'omero prendo il valore più in basso della scapola e quello più in alto dell'omero
                        # (le slice sono al contrario)

                        max_hor = np.max(shoulder_seg_humerus_data[:,:,pos_array_prof_scap[0]:],axis=1) # Asse orizzontale
                        pos_array_hor = np.diff(max_hor,prepend=0,axis=0).nonzero()[0]
                        #*********** AGGIUNTA *************#
    
                        index_split = np.where(np.diff(pos_array_hor,prepend=0)[1:] > limit)[0]
                        lista_split = np.split(pos_array_hor,index_split+1) 
                        cur_width = 0
                        for el in lista_split:
                            if el[-1]-el[0] > cur_width:
                                pos_array_hor = el # Sostituisco direttamente pos_array_hor
                                cur_width = el[-1]-el[0]
                        #************* FINE ***************#
                        pos_array_hor_ct.append(pos_array_hor[0])
                        pos_array_hor_ct.append(pos_array_hor[-1])

                        max_ver = np.max(shoulder_seg_humerus_data[:,:,pos_array_prof_scap[0]:],axis=0) # Asse verticale
                        pos_array_ver = np.diff(max_ver,prepend=0,axis=0).nonzero()[0]
                        #*********** AGGIUNTA *************#
                        index_split = np.where(np.diff(pos_array_ver,prepend=0)[1:] > limit)[0]
                        lista_split = np.split(pos_array_ver,index_split+1) 
                        cur_height = 0
                        for el in lista_split:
                            if el[-1]-el[0] > cur_height:
                                pos_array_ver = el
                                cur_height = el[-1]-el[0]
                        #************* FINE ***************#
                        pos_array_ver_ct.append(pos_array_ver[0])
                        pos_array_ver_ct.append(pos_array_ver[-1])

                        # cur_width = pos_array_hor[-1] - pos_array_hor[0]
                        if cur_width > max_width:
                            max_width = cur_width

                        # cur_height = pos_array_ver[-1] - pos_array_ver[0]
                        if cur_height > max_height:
                            max_height = cur_height

                        cur_slices = pos_array_prof_hum[-1] - pos_array_prof_scap[0]
                        if cur_slices > max_slices:
                            max_slices = cur_slices
                    else:
                        logger.info(f"Ct {i} spalla sinistra non segmentata o troppo piccola")
                if (right_humerus_label in unique_labels) and (right_scapula_label in unique_labels): # Spalla destra
                    shoulder_seg_humerus_data = np.where(shoulder_seg_data == right_humerus_label,1,0).astype(np.uint8)
                    shoulder_seg_scapula_data = np.where(shoulder_seg_data == right_scapula_label,1,0).astype(np.uint8)

                    max_prof_hum = np.max(shoulder_seg_humerus_data,axis=(0,1)) # Asse z
                    pos_array_prof_hum = np.diff(max_prof_hum,prepend=0).nonzero()[0] 

                    max_prof_scap = np.max(shoulder_seg_scapula_data,axis=(0,1)) # Asse z
                    pos_array_prof_scap = np.diff(max_prof_scap,prepend=0).nonzero()[0]

                    if pos_array_prof_scap.size != 0 and pos_array_prof_hum[-1]-pos_array_prof_scap[0] >= limit_slices:
                        logger.info(f"Processo ct {i}: spalla destra")
                        pos_array_prof_ct.append(pos_array_prof_scap[0])
                        pos_array_prof_ct.append(pos_array_prof_hum[-1]) 

                        max_hor = np.max(shoulder_seg_humerus_data[:,:,pos_array_prof_scap[0]:],axis=1) # Asse orizzontale
                        pos_array_hor = np.diff(max_hor,prepend=0,axis=0).nonzero()[0]
                        #*********** AGGIUNTA *************#
                        index_split = np.where(np.diff(pos_array_hor,prepend=0)[1:] > limit)[0]
                        lista_split = np.split(pos_array_hor,index_split+1) 
                        cur_width = 0
                        for el in lista_split:
                            if el[-1]-el[0] > cur_width:
                                pos_array_hor = el # Sostituisco direttamente questo
                                cur_width = el[-1]-el[0]
                        #************* FINE ***************#
                        pos_array_hor_ct.append(pos_array_hor[0])
                        pos_array_hor_ct.append(pos_array_hor[-1])

                        max_ver = np.max(shoulder_seg_humerus_data[:,:,pos_array_prof_scap[0]:],axis=0) # Asse verticale
                        pos_array_ver = np.diff(max_ver,prepend=0,axis=0).nonzero()[0]
                        #*********** AGGIUNTA *************#
                        index_split = np.where(np.diff(pos_array_ver,prepend=0)[1:] > limit)[0]
                        lista_split = np.split(pos_array_ver,index_split+1) 
                        cur_height = 0
                        for el in lista_split:
                            if el[-1]-el[0] > cur_height:
                                pos_array_ver = el
                                cur_height = el[-1]-el[0]
                        #************* FINE ***************#
                        pos_array_ver_ct.append(pos_array_ver[0])
                        pos_array_ver_ct.append(pos_array_ver[-1])

                        #cur_width = pos_array_hor[-1] - pos_array_hor[0]
                        if cur_width > max_width:
                            max_width = cur_width

                        #cur_height = pos_array_ver[-1] - pos_array_ver[0]
                        if cur_height > max_height:
                            max_height = cur_height

                        cur_slices = pos_array_prof_hum[-1] - pos_array_prof_scap[0]
                        if cur_slices > max_slices:
                            max_slices = cur_slices
                    else:
                        logger.info(f"Ct {i} spalla destra non segmentata o troppo piccola")            
            else:
                logger.info(f"Ct {i} esclusa")
        else:
            logger.info(f"Ct {i} non esiste")
        pos_array_hor_all.append(pos_array_hor_ct)
        pos_array_ver_all.append(pos_array_ver_ct)
        pos_array_prof_all.append(pos_array_prof_ct)
            
    return pos_array_hor_all,pos_array_ver_all,pos_array_prof_all,max_width,max_height,max_slices    

if __name__ == "__main__":
    st = time.time()
    logger.remove()
    logger.add("humerus_preprocessing/largest_box_humerus_res_log.log", format="{time:DD-MM HH:mm:ss} - {message}", level="INFO")
    pos_array_hor_all,pos_array_ver_all,pos_array_prof_all,max_width,max_height,max_slices = find_largest_box_humerus()
    logger.info(f"Tempo: {(time.time()-st):.2f} sec")

    with open(f"humerus_preprocessing/arrays/pos_array_hor_all.pkl","wb") as f_hor,open(f"humerus_preprocessing/arrays/pos_array_ver_all.pkl","wb") as f_ver,open(f"humerus_preprocessing/arrays/pos_array_prof_all.pkl","wb") as f_prof:
        pickle.dump(pos_array_hor_all,f_hor)
        pickle.dump(pos_array_ver_all,f_ver)
        pickle.dump(pos_array_prof_all,f_prof)