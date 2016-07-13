# Written by: Erick Cobos T
# Date: 12-July-2016
"""Writes all images in the stimulus to a folder (wasteful but easier)"""

import numpy as np
import h5py
import scipy.misc

# Set params
folder_name = 'images/' # Needs to exist beforehand

# Open matlab-style file
stim_file = h5py.File('Stimuli.mat','r')
stim = np.array(stim_file['st'])
#TODO: Make sure the last index is the num_images and that the first three are width, height, color. Maybe print a couple first to make sure.

# Save images
num_images = stim.shape[3]
for i in range(num_images):
	save_path = folder_name + 'img_' + str(i+1) + '.png'
	scipy.misc(save_path, stim[..., i])
