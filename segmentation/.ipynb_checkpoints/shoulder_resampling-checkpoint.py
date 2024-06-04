import os
import numpy as np
import nibabel as nib
import dicom2nifti
import pickle
from totalsegmentator.python_api import totalsegmentator
from loguru import logger
import sys
import resampling

def shoulder_resampling(excluded_cts,spacing=(0.5,0.5,0.5),int_order=2):
    """
    Effettua il resampling di tutte le spalle in modo da ottenere immagini isotropiche
    """
    for i in range(1,1251):
        if i not in excluded_cts:
            try:
                if os.path.exists(f"processing/{i}/shoulder_nifti.nii"):
                    nifti_img = nib.load(f"processing/{i}/shoulder_nifti.nii")
                    nifti_img_res = resampling.change_spacing(nifti_img,new_spacing=spacing,order=int_order,dtype="<i2")
                    nib.save(nifti_img_res,f"processing/{i}/shoulder_nifti_res.nii") 
                    logger.info(f"Resampling spalla {i} eseguito")
                else:
                    logger.info(f"Spalla {i} non esistente")
            except:
                logger.warning(f"Eccezione spalla {i}")
        else:
            logger.info(f"Spalla {i} esclusa")

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="WARNING")
    logger.add("segmentation/resampling_log.log", format="{time:DD-MM HH:mm:ss} - {message}", level="INFO")

    with open("processing/excluded_cts.pkl","rb") as f:
        excluded_cts = pickle.load(f)
    shoulder_resampling(excluded_cts)