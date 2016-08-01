# Written by: Erick Cobos T
# Date: 01-August-2016
""" Generates feature representations for every second of video using the video
frames' image vectors.

The first representation just subsamples and takes the image vector of the frame
at the given timestep and the second averages over all frames in the previous
second.
"""
import h5py
import numpy as np

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

print('Done!')
