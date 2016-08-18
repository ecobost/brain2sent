# Written by: Erick Cobos T
# Date: 18-August-2016
"""Compute and print test metrics for all models. """
import numpy as np
import h5py
from scipy.stats import gamma
from numpy.fft import fft, ifft
from sklearn import metrics

def main:
	"""Main function. Tests all models."""
	
	# Lasso models (each takes around ... days)
	test_linear('l1_4sec', 'test_4sec_bold.h5', 'test_feats.h5')
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


def read_inputs(features_filename, targets_filename, var_prefix=''):
	# Read regression features
	with h5py.File(features_filename, 'r') as Xs_file:
		Xs = np.array(Xs_file[var_prefix + 'responses'])

	# Read regression targets
	with h5py.File(targets_filename, 'r') as y_file:
		y = np.array(y_file[var_prefix + 'feats'])
	
	return Xs, y
	
def estimate_SNR(signal, original):
	""" Estimate the signal-to-noise ratio needed for deconvolution."""
	# Set params
	SNRs = np.logspace(-2, 1.5, 50)
	
	# Deconvolve and compute correlations for each SNR, signal pair
	results = np.zeros((len(SNRs), signal.shape[1]))
	for i, SNR in enumerate(SNRs):
		deconvolved = deconvolve(signal, SNR)
		results[i,:] = column_wise_corr(original, deconvolved)
	
	# Return best SNR for each signal
	return SNRs[results.argmax(axis=0)]

def deconvolve(signal, SNR):
	"""Deconvolve signal using Wiener deconvolution. SNR estimated in the 
	training set."""

	# If SNR is scalar, use the same for all signals
	if np.isscalar(SNR) and signal.ndim > 1:
		SNR = np.full(signal.shape[1], SNR)

	# Wiener deconvolution
	H = fft(hrf, len(signal))
	wiener_filter = np.expand_dims(np.conj(H), -1) / np.add.outer(H * np.conj(H), 
															 	  1 / SNR**2)
	deconvolved = np.real(ifft(wiener_filter * fft(signal)))

	return deconvolved

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
	
def compute_metrics(y_true, y_pred):
	""" Compute RMSE, correlation and R^2 scores. """
	# RMSE
	rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

	# Correlation
	r_feature = column_wise_corr(y_true, y_pred).mean()
	r_sample = row_wise_corr(y_true, y_pred).mean()

	# R2
	r2_feature = metrics.r2_score(y_true, y_pred, multioutput='uniform_average')
	r2_sample = metrics.r2_score(y_true.transpose(), y_pred.transpose(),
								 multioutput='uniform_average')

	return rmse, r_feature, r_sample, r2_feature, r2_sample

def column_wise_corr(A, B):
	""" Computes the column-wise correlation between A and B."""
	return row_wise_corr(A.transpose(), B. transpose())

def row_wise_corr(A, B):
	""" Computes row-wise correlation between A and B."""
	corr = []
	for x, y in zip(A, B):
		corr.append(np.corrcoef(x,y)[0,1])
	return np.array(corr)

def test_linear(model_filename, features_filename, targets_filename):
	""" Tests a linear model."""
	print('Testing model', model_filename[:-3])

	# Read features and targets
	X_test, y_test = read_inputs(features_filename, targets_filename, 'test_')
	
	# Read model
	with h5py.File(model_filename, 'r') as model_file:
		weights = np.array(model_file['weights'])
		bias = np.array(model_file['bias'])
	
	# Make predictions
	y_pred = np.dot(X_test, weights.transpose()) + bias

	# Make predictions in the training set (to rescale test set predictions)
	X_train, y_train = read_inputs('train' + features_filename[4:],
								   'train' + targets_filename[4:])
	y_train_pred = np.dot(X_train, weights.transpose()) + bias

	# If the model predicts convolved features, deconvolve them
	if '_conv_' in model_filename:
		SNR = estimate_SNR(y_train_pred, y_train)
		y_pred = deconvolve(y_pred, SNR)
		y_train_pred = deconvolve(y_train_pred, SNR)

	# Drop first and last 7 predictions to avoid convolution/smoothing artifacts
	y_test = y_test[7:-7, :]
	y_pred = y_pred[7:-7, :]
	
	# Compute set of metrics (before rescaling)
	metrics = compute_metrics(y_test, y_pred)

	# Report
	print('RMSE, r_feature, r_sample, R^2_feature, R^2_sample')
	print(metrics)


	# Rescale y_pred so each feature has the same spread as the training targets
	expected_mean = y_train_pred.mean()
	expected_std = y_train_pred.std()
	desired_mean = y_train.mean() 
	desired_std = y_train.std()
	
	y_pred = (y_pred - expected_mean)/expected_std # normalize to unit std
	y_pred = (y_pred * desired_std) + desired_mean # give new spread
	y_pred[y_pred < 0] = 0 # set any negative prediction to zero
	
	# Compute metrics
	metrics = compute_metrics(y_test, y_pred)
	print('After rescaling...')
	print('RMSE, r_feature, r_sample, R^2_feature, R^2_sample')
	print(metrics)

def test_neural_network():
	""" Test the neural network models. """
	print('Testing model', model_filename[:-3])

	# Read features and targets
	X_test, y_test = read_inputs(features_filename, targets_filename, 'test_')
	
	# Read model
	with h5py.File(model_filename, 'r') as model_file:
		weights_ih = np.array(model_file['weights_ih'])
		bias_ih = np.array(model_file['bias_ih'])
		weights_ho = np.array(model_file['weights_ho'])
		bias_ho = np.array(model_file['bias_ho'])
	
	# Make predictions
	hidden = np.dot(X_test, weights_ih.transpose()) + bias_ih
	y_pred = np.dot(hidden, weights_ho.transpose()) + bias_ho

	# Make predictions in the training set (to rescale test set predictions)
	X_train, y_train = read_inputs('train' + features_filename[4:],
								   'train' + targets_filename[4:])
	hidden = np.dot(X_train, weights_ih.transpose()) + bias_ih
	y_train_pred = np.dot(hidden, weights_ho.transpose()) + bias_ho

	# If the model predicts convolved features, deconvolve them
	if '_conv_' in model_filename:
		SNR = estimate_SNR(y_train_pred, y_train)
		y_pred = deconvolve(y_pred, SNR)
		y_train_pred = deconvolve(y_train_pred, SNR)

	# Drop first and last 7 predictions to avoid convolution/smoothing artifacts
	y_test = y_test[7:-7, :]
	y_pred = y_pred[7:-7, :]
	
	# Compute set of metrics (before rescaling)
	metrics = compute_metrics(y_test, y_pred)

	# Report
	print('RMSE, r_feature, r_sample, R^2_feature, R^2_sample')
	print(metrics)


	# Rescale y_pred so each feature has the same spread as the training targets
	expected_mean = y_train_pred.mean()
	expected_std = y_train_pred.std()
	desired_mean = y_train.mean() 
	desired_std = y_train.std()
	
	y_pred = (y_pred - expected_mean)/expected_std # normalize to unit std
	y_pred = (y_pred * desired_std) + desired_mean # give new spread
	y_pred[y_pred < 0] = 0 # set any negative prediction to zero
	
	# Compute metrics
	metrics = compute_metrics(y_test, y_pred)
	print('After rescaling...')
	print('RMSE, r_feature, r_sample, R^2_feature, R^2_sample')
	print(metrics)

if name == "__main__":
	main()
