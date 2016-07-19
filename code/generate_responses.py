# Written by: Erick Cobos T (a01184587@itesm.mx)
# Date: 17-July-2016
""" Generates different brain activation matrices (in h5 files) from BOLD data.

Reads matrices from Gallant's data, creates the brain mask, drops invalid
columns and generates a matrix (timesteps x voxels) for (1) BOLD responses, 
(2) BOLD responses at different time delays (shifted by 4, 5, 6, 7 and 8 secs)
and (3) estimated neural responses. See code for details.

Works for train or test responses.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import nipy.modalities.fmri.hemodynamic_models as hm

# Read BOLD data file
voxel_file = h5py.File('VoxelResponses_subject1.mat', 'r')
rois = voxel_file['roi']
bold = voxel_file['rt'] # 'rv' for test responses

# Extract ROIs (later used as mask)
brain = np.zeros([18, 64, 64])
i = 1
for roi in rois:
	brain = brain + 8*i*np.array(rois[roi])
	i += 8

# Save ROIs as images (to check against mean EPI and visualization)
for i in range(18):
	plt.imsave('coronal_' + str(i+1) + '.png', brain[i,...])

# Create mean EPI
bold = np.array(bold).transpose()
epi_mean = bold.mean(axis = 0)
epi_mean = epi_mean.reshape([18, 64, 64])
for i in range(18):
	plt.imsave('mean_' + str(i+1) + '.png', epi_mean[i,...])

# Apply ROI mask
roi_mask = (brain > 0).flatten()
bold = bold[:, roi_mask]

# Clean data (delete columns with any nans)
nan_cols = np.isnan(bold).any(axis=0)
nan_mask = np.logical_not(nan_cols) 
bold = bold[:, nan_mask] # 8982 remaining voxels for S1
# Tested: Same voxels are dropped for test responses, i.e., masks are the same

# Save as BOLD
bold_file = h5py.File('bold.h5', 'w')
bold_file.create_dataset('responses', data=bold)
bold_file.close()

# Save as BOLD with different delays (from 4-8 secs)
for delay in [4, 5, 6, 7, 8]:
	delay_bold = np.full(bold.shape, np.nan, dtype = 'float32')
	delay_bold[delay:, :] = bold[:-delay, :]
	delay_file = h5py.File('bold_' + str(delay) + 's_delay.h5', 'w')
	delay_file.create_dataset('responses', data=delay_bold)
	delay_file.close()

# Save as deconv BOLD
deconv_bold = deconvolve_bold(bold) # deconvolve via polynomial division
deconv_file = h5py.File('deconv.h5', 'w')
deconv_file.create_dataset('responses', data=deconv_bold)
deconv_file.close()

# Close stimuli file
voxel_file.close()

def deconvolve_bold(bold):
""" Deconvolving the neural response via polydiv using the canonical HRF.

"If u and v are vectors of polynomial coefficients, convolving them is 
equivalent to multiplying the two polynomials, and deconvolution is polynomial
division." (Explanation from Matlab's deconv() function)

This is probably the simplest approach to deconvolution. See Wu et al., 2013 and Pedregosa et al., 2015 for better ways to do it: they estimate an HRF and neural
responses per voxel via GLM (both provide code).

Args:
	bold: A matrix (timesteps x voxels). BOLD activations.

Returns:
	deconv_bold: A matrix (timesteps x voxels). The (normalized) estimated 
		neural responses that produced the input activations.
"""
deconv_bold = np.zeros_like(bold) # output matrix
hrf = hm.glover_hrf(1, oversampling=1) # 32-second canonical HRF
for voxel in range(bold.shape[1]):
	# pad input by filter_size-1 to recover the entire response vector 
	padded_bold = np.pad(bold[:, voxel], (0, 31), 'constant', constant_values=0)
	
	# drop leading zeros in HRF and deconvolve
	response, _ = np.polydiv(padded_bold, hrf[4:]) 
	
	# realign and store neural responses
	deconv_bold[:, voxel] = response[4:]

# normalize per voxel
deconv_mean = deconv_bold.mean(axis=0)
deconv_std = deconv_bold.std(axis=0)
deconv_bold = (deconv_bold - deconv_mean)/deconv_std

return deconv_bold
