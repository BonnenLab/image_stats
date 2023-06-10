import pyrtools as pt
import cv2
import numpy as np
import os
import sys
from scipy.signal import fftconvolve
from math import pi

DATA_PATH = sys.argv[1]
OUTFILE = sys.argv[2]

imgs = os.listdir(DATA_PATH)

# create kernels
gp = np.array([0.037659, 0.249153, 0.426375, 0.249153, 0.037659])
gp = gp.reshape((1,5))
gd = np.array([-0.109604, -0.276691, 0.000000, 0.276691, 0.109604])
gd = gd.reshape((1,5))
bfilt = np.array([1, 2, 1])/4
bfilt = bfilt.reshape(3, 1)

for img in imgs:
    # read image
    img = cv2.imread(DATA_PATH + img)
    # convert to greyscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # normalize
    img = img / 255.0
    # get image pyramid
    Gpyr = pt.pyramids.GaussianPyramid(img,6)

    all_hists = [] # list of the 6 histograms for the current frame
    for j in range(6):
        # derivative in x direction
        Ix = pt.pyramids.c.wrapper.corrDn(pt.pyramids.c.wrapper.corrDn(Gpyr.pyr_coeffs[j, 0], gd), gp.T)
        # derivative in y direction
        Iy = pt.pyramids.c.wrapper.corrDn(pt.pyramids.c.wrapper.corrDn(Gpyr.pyr_coeffs[j, 0], gp), gd.T)
    
        Mxx = fftconvolve(fftconvolve(Ix*Ix, bfilt, 'valid'), bfilt.T, 'valid')
        Myy = fftconvolve(fftconvolve(Iy*Iy, bfilt, 'valid'), bfilt.T, 'valid')
        Mxy = fftconvolve(fftconvolve(Ix*Iy, bfilt, 'valid'), bfilt.T, 'valid');
    
        term1   = (Mxx + Myy)/2
        term2   = (term1**2 - (Mxx*Myy - Mxy**2)) ** .5
    
        eps = 2.2204e-16
    
        ev1 = term1 + term2
        ev2 = term1 - term2
        ori = np.arctan2(Mxx-ev2, Mxy)-pi/2
        energy = ev1 + ev2
        orientedness  =  ((ev1-ev2)/(ev1+ev2+eps)) **2

        height, width = orientedness.shape[:2]
        
        # create circle mask to prevent artifacts
        rad = .9*height/2
        x = np.linspace(-width/2, width/2-1, width)
        y = np.linspace(-height/2, height/2-1, height)
        xv, yv = np.meshgrid(x, y)
        ind_central = (xv**2 + yv**2)**.5 > rad
    
        ind1 = energy < max(np.quantile(energy[:],.68), 1e-4)
        ind2 = orientedness < .8
        ori_thresholded = ori
        ori_thresholded[ind1] = np.nan
        ori_thresholded[ind2] = np.nan
        ori_thresholded[ind_central] = np.nan
    
        ori_deg = ori_thresholded.flatten() * (180/pi)
        
        # bin orientations
        h = np.histogram(ori_deg, bins=35,range=(-87.5,87.5))
        
        ind_circ_lower = ori_deg < -87.5
        ind_circ_upper = ori_deg > 87.5 
    
        fs = list(h[0])
        fs.insert(0, np.sum(ori_deg[ind_circ_lower]/ori_deg[ind_circ_lower]) + np.sum(ori_deg[ind_circ_upper]/ori_deg[ind_circ_upper]))
        fs.append(np.sum(ori_deg[ind_circ_lower]/ori_deg[ind_circ_lower]) + np.sum(ori_deg[ind_circ_upper]/ori_deg[ind_circ_upper]))
        fs = np.array(fs)
        all_hists.append(fs)

    np.save(OUTFILE+str(imgs.index(img))+'.npy', np.array(all_hists))


