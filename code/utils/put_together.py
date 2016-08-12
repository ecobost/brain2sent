# Written by: Erick Cobos T. (a01184587@itesm.mx)
# Date: 11-Aug-2016
"""Little script to join the output of two semi-trained models."""
import h5py
import numpy as np

previous_model = 'l2_5sec_500_v1.h5'
new_model = 'l2_5sec_700.h5'
preserve_old = 500

v1 = h5py.File(previous_model, 'r')
weights1 = np.array(v1['weights'])
bias1 = np.array(v1['bias'])
v1.close()

v2 = h5py.File(new_model, 'r')
weights2 = np.array(v2['weights'])
bias2 = np.array(v2['bias'])
v2.close()

bias2[:preserve_old] = bias1[:preserve_old]
weights2[:preserve_old, :] = weights1[:preserve_old, :]

model_file = h5py.File('v3' + new_model, 'w')
model_file.create_dataset('weights', data=weights2)
model_file.create_dataset('bias', data=bias2)
model_file.close()
