# Written by: Erick Cobos T
# Date: 12-July-2016
"""Writes all images in the stimulus to a folder (wasteful but easier).
Example:
	$ python3 create_stim_images.py
"""

import numpy as np
import h5py
import scipy.misc

# Set params
folder_name = 'images/' # Needs to exist beforehand

# Open matlab-style file
print('Reading input...')
stim_file = h5py.File('Stimuli.mat','r')
stim = np.array(stim_file['st']) # Change to 'sv' for validation stimuli
stim = stim.transpose()

# Save images
print('Writing images...')
num_images = stim.shape[3]
for i in range(num_images):
	save_path = folder_name + 'img_' + str(i+1) + '.png'
	scipy.misc.imsave(save_path, stim[..., i])

# Done
print('Done!', num_images, 'images written')
