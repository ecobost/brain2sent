# Written by: Erick Cobos T (a01184587@itesm.mx)
# Date: 04-August-2016
""" Extract BOLD and ROI info and save as h5 files.

Reads matrices from Gallant's data, creates the brain mask, drops invalid
columns and generates a matrix (timesteps x voxels) for (1) original BOLD
activations, (2) BOLD activations at different delays (4,5,6 and 7 secs) and
(3) their temporally smoothed versions (1-d gaussian filter with sigma = 1.5 
seconds). It also saves the ROI mask.

Works for train or test responses.
"""
import h5py
import numpy as np
from scipy import ndimage

# Set params
rt_or_rv = 'rt' # change to rv for test responses
file_prefix = 'train_' # whether to write train or test files

# Read BOLD data file
voxel_file = h5py.File('VoxelResponses_subject1.mat', 'r')
rois = voxel_file['roi']
bold = voxel_file[rt_or_rv]

# Extract ROIs
brain = np.zeros([18, 64, 64])
roi_id = 1
for roi in rois:
	brain = brain + roi_id*np.array(rois[roi])
	print(roi, roi_id)
	roi_id += 1
	
# Extract BOLD
bold = np.array(bold).transpose()

# Select voxels from all regions
roi_mask = (brain > 0).flatten()
bold = bold[:, roi_mask]

# Clean data (delete columns with any nans)
nan_cols = np.isnan(bold).any(axis=0)
nan_mask = np.logical_not(nan_cols) 
bold = bold[:, nan_mask] # 8982 remaining voxels for S1
# Tested: Same voxels are dropped for train and test responses

# Save as BOLD
with h5py.File(file_prefix + 'bold.h5', 'w') as bold_file:
	bold_file.create_dataset('responses', data=bold)

# Create smooth BOLD
smooth_bold = ndimage.gaussian_filter1d(bold, sigma=1.5, axis=0)

# Save as smooth BOLD
with h5py.File(file_prefix + 'smooth_bold.h5', 'w') as smooth_bold_file:
	smooth_bold_file.create_dataset('responses', data=smooth_bold)

# Save as BOLD with different delays (from 4-7 secs)
for delay in [4, 5, 6, 7]:
	delay_bold = np.full(bold.shape, np.nan, dtype='float32')
	delay_bold[:-delay, :] = bold[delay:, :]
	with h5py.File(file_prefix + str(delay) + 'sec_bold.h5', 'w') as delay_file:
		delay_file.create_dataset('responses', data=delay_bold)
	
# Save as smooth BOLD with different delays (from 4-7 secs)
for delay in [4, 5, 6, 7]:
	delay_bold = np.full(smooth_bold.shape, np.nan, dtype='float32')
	delay_bold[:-delay, :] = smooth_bold[delay:, :]
	with h5py.File(file_prefix + str(delay) + 'sec_smooth_bold.h5', 'w') as delay_file:
		delay_file.create_dataset('responses', data=delay_bold)

# Create ROI per voxel (later used for voxel selection)
roi_info = (brain[brain > 0])[nan_mask]

# Save ROI info
with h5py.File('roi_info.h5', 'w') as roi_file:
	roi_file.create_dataset('rois', data=roi_info)

# Close voxel file
voxel_file.close()

print('Done!')



### Optionally,
import matplotlib.pyplot as plt

# Save ROIs as images (for visualization)
for i in range(18):
	plt.imsave('coronal_' + str(i+1) + '.png', brain[i,...])
