# Written by: Erick Cobos T
# Date: 04-August-2016
""" Generates feature representations for every second of video using the video
frames' image vectors.

The first representation averages over all frames in the previous second of 
video and the second convolves the features along the time axis with a canonical
HRF.
"""
import h5py
import numpy as np
import nipy.modalities.fmri.hemodynamic_models as hm

# Set params
file_prefix = 'train_' # whether to read/write train or test files

# Read feats
with h5py.File(file_prefix + 'full_feats.h5', 'r') as full_feats_file:
	full_feats = np.array(full_feats_file['feats'])

# Average every 15 rows to create feats
onesec_feats = full_feats.reshape(-1, 15, 768).mean(axis=1)

# Save 1 second feats
with h5py.File(file_prefix + 'feats.h5', 'w') as onesec_feats_file:
	onesec_feats_file.create_dataset('feats', data=onesec_feats)

# Convolve the feature representations with a canonical HRF
hrf = hm.glover_hrf(1, oversampling=15)
full_conv_feats = np.apply_along_axis(lambda x: np.convolve(hrf, x), axis=0,
									  arr=full_feats)
conv_feats = full_conv_feats[14:-479:15, :] # delete end and subsample

# Save convolved feats
with h5py.File(file_prefix + 'conv_feats.h5', 'w') as conv_feats_file:
	conv_feats_file.create_dataset('feats', data=conv_feats)

print('Done!')
