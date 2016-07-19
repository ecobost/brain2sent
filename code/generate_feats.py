# Written by: Erick Cobos T
# Date: 19-July-2016
#TODO: generate_feats
import nipy.modalities.fmri.hemodynamic_models as hm

# Set params
file_prefix = 'train' # Whether to read/write train or test files

# Read feats
feats_file = h5py.File('train_feats.h5', 'r')
feats_data = feats_file['feats']
feats_data = np.array(feats_data)

hm.compute_regressors
