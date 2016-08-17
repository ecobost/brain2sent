# Written by: Erick Cobos T
# Date: 16-August-2016
""" Generates feature representations for every second of video using the video
frames' image vectors.

The first representation averages over all frames in the previous second of 
video and the second convolves the features along the time axis with a canonical
HRF.
"""
import h5py
import numpy as np
from scipy.stats import gamma

# Set params
file_prefix = 'train_' # whether to read/write train or test files

def main():
	# Read feats
	with h5py.File(file_prefix + 'full_feats.h5', 'r') as full_feats_file:
		full_feats = np.array(full_feats_file['feats'])

	# Average every 15 rows to create feats
	onesec_feats = full_feats.reshape(-1, 15, 768).mean(axis=1)

	# Save one second feats
	with h5py.File(file_prefix + 'feats.h5', 'w') as onesec_feats_file:
		onesec_feats_file.create_dataset('feats', data=onesec_feats)

	# Convolve the feature representations with a canonical HRF
	hrf = glover_hrf(np.linspace(0, 32, 481)) # 481 = 32 secs x 15 frames + 1
	full_conv_feats = np.apply_along_axis(lambda x: np.convolve(hrf, x), axis=0,
										  arr=full_feats)
	conv_feats = full_conv_feats[7:108000:15, :] # subsample and delete end

	# Save convolved feats
	with h5py.File(file_prefix + 'conv_feats.h5', 'w') as conv_feats_file:
		conv_feats_file.create_dataset('feats', data=conv_feats)

	print('Done!')

def glover_hrf(timepoints):
	""" Canonical (Glover) HRF.
		
		Based on the nipy implementation (github.com/nipy/nipy/blob/master/nipy/
		modalities/fmri/hrf.py). Peaks at 5.4 with 5.2 FWHM, undershoot peak at
		10.8 with FWHM of 7.35 and amplitude of -0.35. 
		
		Args:
		timepoints: Array of floats. Time points (in seconds, onset is 0).
	"""	
	# Compute HRF as a difference of gammas and normalize
	hrf = gamma.pdf(timepoints, 7, scale=0.9) - 0.35 * gamma.pdf(timepoints, 13,
																 scale=0.9)
	hrf /= hrf.sum()
	return hrf
	
if name == "main":
	main()
