# Written by: Erick Cobos T
# Date: 20-July-2016
""" Generates feature representations for every second of video using the image
vectors.

The first representation just subsamples and takes the image vector of the frame
at the given timestep, the second averages over all feature representations in
the previous second and the third convolves the features along the time axis 
with a canonical HRF.
"""

import h5py
import numpy as np
import nipy.modalities.fmri.hemodynamic_models as hm

# Set params
file_prefix = 'train_' # whether to read/write train or test files

# Read feats
full_feats_file = h5py.File(file_prefix + 'full_feats.h5', 'r')
full_feats = full_feats_file['feats']
full_feats = np.array(full_feats)
full_feats_file.close()

# Subsample one every 15 rows to create feats
feats = full_feats[14::15, :]

# Save feats
feats_file = h5py.File(file_prefix + 'feats.h5', 'w')
feats_file.create_dataset('feats', data=feats)
feats_file.close()

# Average every 15 rows to create 1sec_feats
onesec_feats = full_feats.reshape(-1, 15, 768).mean(axis=1)

# Save 1 second feats
onesec_feats_file = h5py.File(file_prefix + '1sec_feats.h5', 'w')
onesec_feats_file.create_dataset('feats', data=onesec_feats)
onesec_feats_file.close()

# Convolve the feature representations with a canonical HRF
hrf = hm.glover_hrf(1, oversampling=15)
full_conv_feats = np.apply_along_axis(lambda x: np.convolve(hrf, x), axis=0,
									  arr=full_feats)
conv_feats = full_conv_feats[14:-479:15, :] # delete end and subsample

# Save convolved feats
conv_feats_file = h5py.File(file_prefix + 'conv_feats.h5', 'w')
conv_feats_file.create_dataset('feats', data=conv_feats)
conv_feats_file.close()
