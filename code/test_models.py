# Written by: Erick Cobos T
# Date: 13-August-2016

def main:
	"""Main function. Tests all models."""
	
	# Lasso models (each takes around ... days)
	test_linear('l1_4sec', f'test_4sec_bold.h5', 'test_feats.h5')
	test_linear('l1_5sec', 'test_5sec_bold.h5', 'test_feats.h5')
	test_linear('l1_6sec', 'test_6sec_bold.h5', 'test_feats.h5')
	test_linear('l1_7sec', 'test_7sec_bold.h5', 'test_feats.h5')
	test_linear('l1_4sec_smooth', 'test_4sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('l1_5sec_smooth', 'test_5sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('l1_6sec_smooth', 'test_6sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('l1_7sec_smooth', 'test_7sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('l1_conv', 'test_bold.h5', 'test_conv_feats.h5')
	test_linear('l1_conv_smooth', 'test_smooth_bold.h5', 'test_conv_feats.h5')
	test_linear('l1_deconv', 'test_deconv.h5', 'test_feats.h5')
		
	# Ridge models (each takes around 4 days)
	test_linear('l2_4sec.h5', 'test_4sec_bold.h5', 'test_feats.h5')
	test_linear('l2_5sec.h5', 'test_5sec_bold.h5', 'test_feats.h5')
	test_linear('l2_6sec.h5', 'test_6sec_bold.h5', 'test_feats.h5')
	test_linear('l2_7sec.h5', 'test_7sec_bold.h5', 'test_feats.h5')
	test_linear('l2_4sec_smooth.h5', 'test_4sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('l2_5sec_smooth.h5', 'test_5sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('l2_6sec_smooth.h5', 'test_6sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('l2_7sec_smooth.h5', 'test_7sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('l2_conv.h5', 'test_bold.h5', 'test_conv_feats.h5')
	test_linear('l2_conv_smooth.h5', 'test_smooth_bold.h5', 'test_conv_feats.h5')
	test_linear('l2_deconv.h5', 'test_deconv.h5', 'test_feats.h5')
		
	# F-statisc models (each takes around a week)
	test_linear('f_4sec.h5', 'test_4sec_bold.h5', 'test_feats.h5')
	test_linear('f_5sec.h5', 'test_5sec_bold.h5', 'test_feats.h5')
	test_linear('f_6sec.h5', 'test_6sec_bold.h5', 'test_feats.h5')
	test_linear('f_7sec.h5', 'test_7sec_bold.h5', 'test_feats.h5')
	test_linear('f_4sec_smooth.h5', 'test_4sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('f_5sec_smooth.h5', 'test_5sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('f_6sec_smooth.h5', 'test_6sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('f_7sec_smooth.h5', 'test_7sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('f_conv.h5', 'test_bold.h5', 'test_conv_feats.h5')
	test_linear('f_conv_smooth.h5', 'test_smooth_bold.h5', 'test_conv_feats.h5')
	test_linear('f_deconv.h5', 'test_deconv.h5', 'test_feats.h5')
		
	# ROI models (each takes around 2 days)
	test_linear('roi_4sec.h5', 'test_4sec_bold.h5', 'test_feats.h5')
	test_linear('roi_5sec.h5', 'test_5sec_bold.h5', 'test_feats.h5')
	test_linear('roi_6sec.h5', 'test_6sec_bold.h5', 'test_feats.h5')
	test_linear('roi_7sec.h5', 'test_7sec_bold.h5', 'test_feats.h5')
	test_linear('roi_4sec_smooth.h5', 'test_4sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('roi_5sec_smooth.h5', 'test_5sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('roi_6sec_smooth.h5', 'test_6sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('roi_7sec_smooth.h5', 'test_7sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('roi_conv.h5', 'test_bold.h5', 'test_conv_feats.h5')
	test_linear('roi_conv_smooth.h5', 'test_smooth_bold.h5', 'test_conv_feats.h5')
	test_linear('roi_deconv.h5', 'test_deconv.h5', 'test_feats.h5')
	
	# Lasso for feature selection models
	test_linear('l1-fs_4sec.h5', 'test_4sec_bold.h5', 'test_feats.h5')
	test_linear('l1-fs_5sec.h5', 'test_5sec_bold.h5', 'test_feats.h5')
	test_linear('l1-fs_6sec.h5', 'test_6sec_bold.h5', 'test_feats.h5')
	test_linear('l1-fs_7sec.h5', 'test_7sec_bold.h5', 'test_feats.h5')
	test_linear('l1-fs_4sec_smooth.h5', 'test_4sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('l1-fs_5sec_smooth.h5', 'test_5sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('l1-fs_6sec_smooth.h5', 'test_6sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('l1-fs_7sec_smooth.h5', 'test_7sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('l1-fs_conv.h5', 'test_bold.h5', 'test_conv_feats.h5')
	test_linear('l1-fs_conv_smooth.h5', 'test_smooth_bold.h5', 'test_conv_feats.h5')
	test_linear('l1-fs_deconv.h5', 'test_deconv.h5', 'test_feats.h5')
	
	# Ridge for feature selection
	test_linear('l2-fs_4sec.h5', 'test_4sec_bold.h5', 'test_feats.h5')
	test_linear('l2-fs_5sec.h5', 'test_5sec_bold.h5', 'test_feats.h5')
	test_linear('l2-fs_6sec.h5', 'test_6sec_bold.h5', 'test_feats.h5')
	test_linear('l2-fs_7sec.h5', 'test_7sec_bold.h5', 'test_feats.h5')
	test_linear('l2-fs_4sec_smooth.h5', 'test_4sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('l2-fs_5sec_smooth.h5', 'test_5sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('l2-fs_6sec_smooth.h5', 'test_6sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('l2-fs_7sec_smooth.h5', 'test_7sec_smooth_bold.h5', 'test_feats.h5')
	test_linear('l2-fs_conv.h5', 'test_bold.h5', 'test_conv_feats.h5')
	test_linear('l2-fs_conv_smooth.h5', 'test_smooth_bold.h5', 'test_conv_feats.h5')
	test_linear('l2-fs_deconv.h5', 'test_deconv.h5', 'test_feats.h5')
	
	# Neural network models
	test_neural_network('nn_4sec.h5', 'test_4sec_bold.h5', 'test_feats.h5')
	test_neural_network('nn_5sec.h5', 'test_5sec_bold.h5', 'test_feats.h5')
	test_neural_network('nn_6sec.h5', 'test_6sec_bold.h5', 'test_feats.h5')
	test_neural_network('nn_7sec.h5', 'test_7sec_bold.h5', 'test_feats.h5')
	test_neural_network('nn_4sec_smooth.h5', 'test_4sec_smooth_bold.h5', 'test_feats.h5')
	test_neural_network('nn_5sec_smooth.h5', 'test_5sec_smooth_bold.h5', 'test_feats.h5')
	test_neural_network('nn_6sec_smooth.h5', 'test_6sec_smooth_bold.h5', 'test_feats.h5')
	test_neural_network('nn_7sec_smooth.h5', 'test_7sec_smooth_bold.h5', 'test_feats.h5')
	test_neural_network('nn_conv.h5', 'test_bold.h5', 'test_conv_feats.h5')
	test_neural_network('nn_conv_smooth.h5', 'test_smooth_bold.h5', 'test_conv_feats.h5')
	test_neural_network('nn_deconv.h5', 'test_deconv.h5', 'test_feats.h5')
	
def read_inputs(features_filename, targets_filename):
	# Read regression features
	with h5py.File(features_filename, 'r') as Xs_file:
		Xs = np.array(Xs_file['test_responses'])

	# Read regression targets
	with h5py.File(targets_filename, 'r') as y_file:
		y = np.array(y_file['test_feats'])
	
	# Drop first and last 10 samples to avoid convolution/smoothing artifacts
	Xs = Xs[10:-10, :]
	y = y[10:-10, :]
	
	return Xs, y
	
def metrics(y_true, y_pred):
	return r2,RMSE, ...

def test_linear(model_filename, features_filename, targets_filename):
	""" Tests a linear model."""
	print('Testing model', model_filename[:-3])
	
	# Read features and targets
	X_test, y_test = read_inputs(features_filename, targets_filename)
	
	# Read model
	with h5py.File(model_filename, 'r') as model_file:
		weights = np.array(model_file['weights'])
		bias = np.array(model_file['bias'])
	
	# Make predictions
	y_pred = np.dot(X_test, weights.transpose()) + bias
	
	# Compute set of metrics
	
	# Print metrics to screen
	
	# Read training targets (used to rescale the outputs)
	with h5py.File('train' + targets_filename[4:], 'r') as y_file:
		y = np.array(y_file['feats'])
		
	# Rescale y_pred so each feature has the same spread as the training targets
	y_pred = 
	
	# Compute metrics 
	
	# Print 
def test_neural_network():

if name == "__main__":
	main()
