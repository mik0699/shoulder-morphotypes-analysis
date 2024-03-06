import os
import numpy as np
import nibabel as nib
import dicom2nifti
from totalsegmentator.python_api import totalsegmentator
from loguru import logger
import sys

def shoulder_segmentation_resampled():
    """
    Segmentazione spalle isotropiche ottenute tramite resampling
    """
    for i in range(1,1251):
        try:
            if os.path.exists(f"processing/{i}/shoulder_nifti_res.nii"):
                totalsegmentator(f"processing/{i}/shoulder_nifti_res.nii",f'processing/{i}/shoulder_seg_res.nii',ml=True,output_type="nifti",quiet=True,roi_subset=["humerus_left","humerus_right","scapula_left","scapula_right"])
                logger.info(f"Segmentazione spalla {i} eseguita")
            else:
                logger.info(f"Spalla {i} non esistente")
        except:
            logger.warning(f"Eccezione spalla {i}")
        
if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="WARNING")
    logger.add("segmentation/segmentations_res_log.log", format="{time:DD-MM HH:mm:ss} - {message}", level="INFO")

    shoulder_segmentation_resampled()