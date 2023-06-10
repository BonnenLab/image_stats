import os
import cv2
import numpy as np
import pandas as pd
import sys


DATA_PATH = sys.argv[1]
OUTFILE = sys.argv[2]
imgs = os.listdir(DATA_PATH)

rms_vals = []
for img in imgs:
    # read image
    img = cv2.imread(DATA_PATH+img, cv2.IMREAD_COLOR)
    # convert to greyscale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # calculate rms contrast
    contrast = np.std(img_grey)
    rms_vals.append(contrast)
    
df = pd.DataFrame({"Filename": imgs, "Contrast": rms_vals})
df.to_csv(OUTFILE)

