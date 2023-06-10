import os
import numpy as np
import cv2
import sys
import pandas as pd


DATA_PATH = sys.argv[1]
OUTFILE = sys.argv[2]

imgs = os.listdir(DATA_PATH)

rms_center = []
rms_upper = []
rms_lower = []
rms_left = []
rms_right = []

def create_mask(image,x,y):
    r = min(image.shape[:2]) / 8

    # Create a mask with a circular shape
    h, w = image.shape
    ygrid, xgrid = np.ogrid[:h, :w]
    mask = ((xgrid - x)**2 + (ygrid - y)**2) <= r**2
    
    return mask

for img in imgs:
    with np.errstate(divide='ignore'):
        # read image
        img = cv2.imread(DATA_PATH + img)
        # convert to greyscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # create masks
        m1 = create_mask(img,320,240)
        m2 = create_mask(img,320,((240-(117/2))/2))
        m3 = create_mask(img,320-((240+(117/2))/2),240)
        m4 = create_mask(img,320+((240+(117/2))/2),240)
        m5 = create_mask(img,320,240+(240-(240-(117/2))/2))
    
        # Apply the mask to the image
        section = img[m1]
        # calculate rms contrast of section
        rms1 = np.std(section)
        
        section = img[m2]
        rms2 = np.std(section)
        
        section = img[m3]
        rms3 = np.std(section)
        
        section = img[m4]
        rms4 = np.std(section)
        
        section = img[m5]
        rms5 = np.std(section)

        rms_center.append(rms1)
        rms_upper.append(rms2)
        rms_lower.append(rms5)
        rms_left.append(rms3)
        rms_right.append(rms4)

        
df = pd.DataFrame({"Center RMS": rms_center, "Upper RMS": rms_upper, "Lower RMS": rms_lower, "Left RMS": rms_left, "Right RMS": rms_right, "Filename": imgs})
df.to_csv(OUTFILE)

