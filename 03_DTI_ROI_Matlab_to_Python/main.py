# Mancy Chen 25/04/2023
# Convert the Matlab code from Cheima
import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt

# Set the main directory
maindir = '.../Matlab_ROI_to_Python/stats/'
Output = '.../Matlab_ROI_to_Python/Output/'
# Set the FSL directory
fsldir = '.../fsl-6.0.3/data/atlases/JHU/'

# Load the tracts, mask, and mean FA images
#tractsfile = os.path.join(fsldir, 'JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz')
tractsfile = os.path.join(fsldir, 'JHU-ICBM-DWI-1mm.nii.gz')
t = nib.load(tractsfile)
tracts = t.get_fdata()

m = nib.load(os.path.join(maindir, 'mean_FA_skeleton_mask.nii.gz'))
mask = m.get_fdata()

mfa = nib.load(os.path.join(maindir, 'mean_FA.nii.gz'))
meanfa = mfa.get_fdata()

# Set the ROI IDs and tags
#IDs = [1, 2, 9]
IDs = [0, 1, 8]
tags = ['ATR_L', 'ATR_R', 'splenium']

# Select DTI-measure and calculate mean values in tracts of interest
files = [os.path.join(maindir, f) for f in os.listdir(maindir) if f.endswith('_skeletonised.nii.gz')]

for file in files:
    _, name = os.path.splitext(os.path.basename(file))
    _, name = os.path.splitext(name)  # remove .nii

    v = nib.load(file)
    fa = v.get_fdata()
    # fa = fa[fa != 1] = 0
    sz = fa.shape
    nsubj = sz[3]

    fa = fa.reshape(-1, nsubj)

    meas_roi = np.zeros((nsubj, len(IDs)))
    for ii in range(len(IDs)):
        roi = binary_dilation(tracts == IDs[ii]).astype(np.float64)
        roi *= mask
        meas_roi[:, ii] = np.mean(fa[roi > 0, :], axis=0)

    savename = os.path.join(Output, f'tracts_mean_{name}.txt')
    data = dict(zip([''] + tags, np.vstack([tags, meas_roi])))
    T = pd.DataFrame(data)
    T.to_csv(savename, sep='\t', index=False)

# Plot the ROIs
roi = np.zeros_like(tracts)
for iROI in range(len(IDs)):
    thisroi = binary_dilation(tracts == IDs[iROI]).astype(np.float64)
    roi += thisroi * (iROI + 1)

cl = plt.rcParams['axes.prop_cycle'].by_key()['color']
cl.insert(0, (1, 1, 1))
out = roi * mask + mask
sz = out.shape
out = out.reshape(-1)
outcl = np.zeros((len(out), 3))
outcl[out > 0, :] = cl[out[out > 0] - 1]
outcl[~out, :] = meanfa[~out].reshape(-1, 1) @ np.ones((1, 3))
outcl = outcl.reshape((*sz, 3))
outcl = np.transpose(outcl, (2, 0, 1))
outcl = outcl[..., 45:70:4, :]
plt.imshow(outcl.transpose(1, 2, 0))
plt.axis('off')
plt.show()

# Plot the legend
fig, ax = plt.subplots()
ax.plot(np.zeros((20, len(tags) + 1)), linewidth=2)
ax.set_prop_cycle('color', cl)
ax.legend([''] + tags)
plt.show()
