# -*- coding: utf-8 -*-
"""metroChallenge

Evaluation entry-point required by the challenge organisers.  The script:

1. Sets *BASE_DIR* / *base_dir* environment variables so that YAML
   interpolations work after unpacking to an arbitrary directory.
2. Launches the project's Demo pipeline by calling ``run.py`` with Hydra
   overrides (`mode=demo`, `model.name=demo`, etc.).  The Demo pipeline must
   generate a file named **teamsNN.mat** in the project root that contains the
   recognition results (variable **BD**).
3. Once the Demo pipeline finishes, the official evaluation routine
   ``evaluationV2.py`` is executed to compute metrics.

The grading team only needs to:
    1. Unzip the package.
    2. Copy *164* test images to *BD_CHALLENGE/*
    3. Copy *GTCHALLENGETEST.mat* to repo root.
    4. Run ``python metroChallenge.py``.
"""
import numpy as np
import os
import re
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.io as sio
from evaluationV2 import evaluation,compareTestandRef
import subprocess
import sys

# Define directories ==========================================================

# Define your resizing parameter===============================================
#resize_factor = 1


# Read the challenge directory ================================================

#image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
#pattern = re.compile(r'\((\d+)\)')
#numbered_images = []

#for filename in os.listdir(challengeDirectory):
#    if filename.lower().endswith(image_extensions):
#        match = pattern.search(filename)
#        if match:
#            number = int(match.group(1))
#            numbered_images.append((number, filename))

#numbered_images.sort()
#imageFilesList = [filename for filename in numbered_images]

    
# Process all images and save the results in a mat file =======================

#num_images = len(imageFilesList)
#BD = []

#for n_val in range(num_images):
    
    # LOAD IMAGE ------------------------------
#    nom = imageFilesList[n_val][1]
#    print(f"----- {nom} -----")
#    im_path = os.path.join(challengeDirectory, nom)
#    im = np.array(Image.open(im_path).convert('RGB')) / 255.0
    
    
    # fig = plt.figure(figsize=(45,15))
    # plt.imshow(im)   
    # plt.title(f'{imageFilesList[n_val]}')
    
         
    # PROCESS IMAGE --------------------------- 
#    im_resized,bd = processOneMetroImage(nom,im,imageFilesList[n_val][0],resize_factor)
    
    
    # ADD TO GLOBAL RECOGNITION RESULTS -------
#    BD.extend(bd.tolist())

# Save recognition
#sio.savemat(file_out, {'BD': np.array(BD)})


# # Check reference recognition =================================================
# # compareTestandRef(imageFilesList,challengeDirectory,'GTCHALLENGETEST.mat', file_out, resize_factor)
  
    
# Quantitatve evaluation ======================================================
#evaluation('GTCHALLENGETEST.mat', file_out, resize_factor)  # it cannot work without the generated file of random recognition
                                                            

# -----------------------------------------------------------------------------
# 0.  User-configurable paths
# -----------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
IMAGES_DIR = os.path.join(REPO_ROOT, "BD_CHALLENGE")            
GT_MAT     = "GTCHALLENGETEST.mat"     
OUT_MAT    = "teams25.mat"             
RESIZE     = 1                          

# -----------------------------------------------------------------------------
# 1.  Environment variables for YAML `${base_dir}` interpolation
# -----------------------------------------------------------------------------
os.environ["BASE_DIR"] = REPO_ROOT
os.environ["base_dir"] = REPO_ROOT

# -----------------------------------------------------------------------------
# 2.  Run Demo pipeline via run.py (sub-process keeps Hydra parsing isolated)
# -----------------------------------------------------------------------------
print("[metroChallenge] Launching Demo pipeline …")
cmd = [
    sys.executable,
    "run.py",
    "mode=demo",                # override defaults → demo mode
    f"mode.demo.input_path={IMAGES_DIR}",
    "mode.demo.batch_mode=true",   # process entire directory
    "mode.demo.save_results=false", # Demo pipeline should still write teamsNN.mat
    f"mode.demo.output_path={REPO_ROOT}"
]
subprocess.check_call(cmd)
print("[metroChallenge] Demo pipeline finished ")

# -----------------------------------------------------------------------------
# 3.  Quantitative evaluation
# -----------------------------------------------------------------------------
if not os.path.exists(OUT_MAT):
    raise FileNotFoundError(f"Expected result file '{OUT_MAT}' not found – did Demo pipeline create it?")

print("[metroChallenge] Running official evaluation …")
evaluation(GT_MAT, OUT_MAT, RESIZE)
print("[metroChallenge] Evaluation completed.  See scores above.")
