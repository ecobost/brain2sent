# Written by: Erick Cobos T
# Date: 20-August-2016
"""Compute and print test metrics for all models. """
import numpy as np
import h5py
from scipy.stats import gamma
from numpy.fft import fft, ifft

def main():
	"""Main function. Tests all models."""
	# Ridge models
	test_linear('l2_4sec.h5', 'test_4sec_bold.h5')
	test_linear('l2_5sec.h5', 'test_5sec_bold.h5')
	test_linear('l2_6sec.h5', 'test_6sec_bold.h5')
	test_linear('l2_7sec.h5', 'test_7sec_bold.h5')
	test_linear('l2_4sec_smooth.h5', 'test_4sec_smooth_bold.h5')
	test_linear('l2_5sec_smooth.h5', 'test_5sec_smooth_bold.h5')
	test_linear('l2_6sec_smooth.h5', 'test_6sec_smooth_bold.h5')
	test_linear('l2_7sec_smooth.h5', 'test_7sec_smooth_bold.h5')
	test_linear('l2_conv.h5', 'test_bold.h5')
	test_linear('l2_conv_smooth.h5', 'test_smooth_bold.h5')
	test_linear('l2_deconv.h5', 'test_deconv.h5')
		
	# F-statisc models
	test_linear('f_4sec.h5', 'test_4sec_bold.h5')
	test_linear('f_5sec.h5', 'test_5sec_bold.h5')
	test_linear('f_6sec.h5', 'test_6sec_bold.h5')
	test_linear('f_7sec.h5', 'test_7sec_bold.h5')
	test_linear('f_4sec_smooth.h5', 'test_4sec_smooth_bold.h5')
	test_linear('f_5sec_smooth.h5', 'test_5sec_smooth_bold.h5')
	test_linear('f_6sec_smooth.h5', 'test_6sec_smooth_bold.h5')
	test_linear('f_7sec_smooth.h5', 'test_7sec_smooth_bold.h5')
	test_linear('f_conv.h5', 'test_bold.h5')
	test_linear('f_conv_smooth.h5', 'test_smooth_bold.h5')
	test_linear('f_deconv.h5', 'test_deconv.h5')
		
	# ROI models
	test_linear('roi_4sec.h5', 'test_4sec_bold.h5')
	test_linear('roi_5sec.h5', 'test_5sec_bold.h5')
	test_linear('roi_6sec.h5', 'test_6sec_bold.h5')
	test_linear('roi_7sec.h5', 'test_7sec_bold.h5')
	test_linear('roi_4sec_smooth.h5', 'test_4sec_smooth_bold.h5')
	test_linear('roi_5sec_smooth.h5', 'test_5sec_smooth_bold.h5')
	test_linear('roi_6sec_smooth.h5', 'test_6sec_smooth_bold.h5')
	test_linear('roi_7sec_smooth.h5', 'test_7sec_smooth_bold.h5')
	test_linear('roi_conv.h5', 'test_bold.h5')
	test_linear('roi_conv_smooth.h5', 'test_smooth_bold.h5')
	test_linear('roi_deconv.h5', 'test_deconv.h5')
	
	# Neural network models
	test_neural_network('nn_4sec.h5', 'test_4sec_bold.h5')
	test_neural_network('nn_5sec.h5', 'test_5sec_bold.h5')
	test_neural_network('nn_6sec.h5', 'test_6sec_bold.h5')
	test_neural_network('nn_7sec.h5', 'test_7sec_bold.h5')
	test_neural_network('nn_4sec_smooth.h5', 'test_4sec_smooth_bold.h5')
	test_neural_network('nn_5sec_smooth.h5', 'test_5sec_smooth_bold.h5')
	test_neural_network('nn_6sec_smooth.h5', 'test_6sec_smooth_bold.h5')
	test_neural_network('nn_7sec_smooth.h5', 'test_7sec_smooth_bold.h5')
	test_neural_network('nn_conv.h5', 'test_bold.h5')
	test_neural_network('nn_conv_smooth.h5', 'test_smooth_bold.h5')
	test_neural_network('nn_deconv.h5', 'test_deconv.h5')

def read_inputs(features_filename, targets_filename):
	# Read regression features
	with h5py.File(features_filename, 'r') as Xs_file:
		Xs = np.array(Xs_file['responses'])

	# Read regression targets
	with h5py.File(targets_filename, 'r') as y_file:
		y = np.array(y_file['feats'])
		
	# Normalize BOLD data
	Xs = (Xs - Xs[10:-10].mean(axis=0)) / Xs[10:-10].std(axis=0)
	# I shouldn't do this, I should use stats from the train data to normalize 
	# the test data or leave both unnormalized. Problem is that they gave me the
	# normalized training data and unnormalized test data, so I'm forced to 
	# normalize the test data (otherwise the models will produce garbage on the
	# test set) but I don't have the original train data to extract train_mean 
	# and train_std, so I have to use the test stats. I am not happy with this.
	# As this is BOLD data, I'm hoping the test mean and test std give me info
	# about the scanner/session and not about the actual test data. Otherwise,
	# test results would be invalid :(
	
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
	hrf = glover_hrf(np.linspace(0,32,33))
	H = fft(hrf, len(signal), axis=0)
	wiener_filter = np.expand_dims(np.conj(H), -1) / np.add.outer(H * np.conj(H), 
															 	  1 / SNR**2)
	deconvolved = np.real(ifft(wiener_filter * fft(signal, axis=0), axis=0))

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
	
def print_metrics(y_true, y_pred):
	""" Computes and prints RMSE, correlation and R^2 scores. """
	# RMSE
	rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

	# Correlation
	r_feature = column_wise_corr(y_true, y_pred).mean()
	r_sample = row_wise_corr(y_true, y_pred).mean()

	# R2
	r2_feature = column_wise_r2(y_true, y_pred).mean()
	r2_sample = row_wise_r2(y_true, y_pred).mean()
	
	print('|RMSE | r_feature | r_sample | R^2_feature | R^2_sample|')
	print('|', rmse, '|', r_feature, '|', r_sample, '|', r2_feature, '|', 
		  r2_sample, '|')

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
	
def column_wise_r2(y_true, y_pred, min_var=0.001):
	""" Computes R^2 per column."""
	rss = ((y_true-y_pred)**2).sum(axis=0)
	tss = ((y_true- y_true.mean(axis=0))**2).sum(axis=0)
	r2 = 1 - (rss/tss)
	
	# Set (almost) constant columns to zero. 
	r2[y_true.var(axis=0) < min_var] = 0
	# If the variance is close to zero, tss is close to zero and r2 explodes.  
	# These don't say much about performance anyway. Only 649 and 743 fall here.
	
	return r2

def row_wise_r2(y_true, y_pred, min_var=0.001):
	"""Computes R2 per row"""
	return column_wise_r2(y_true.transpose(), y_pred.transpose(), min_var)
	
def test_linear(model_filename, features_filename):
	""" Tests a linear model."""
	print('Testing model', model_filename[:-3])

	# Read features and targets
	X_test, y_test = read_inputs(features_filename, 'test_feats.h5')
	
	# Read model
	with h5py.File(model_filename, 'r') as model_file:
		weights = np.array(model_file['weights'])
		bias = np.array(model_file['bias'])
	
	# Make predictions
	y_pred = np.dot(X_test, weights.transpose()) + bias
	
	# Make predictions on training set (used to estimate SNR and re-scaling)
	X_train, y_train = read_inputs('train' + features_filename[4:], 
								   'train_feats.h5')
	y_train_pred = np.dot(X_train, weights.transpose()) + bias

	# If the model predicts convolved features, deconvolve them
	if '_conv' in model_filename:
		# Estimate the signal-to-noise ratio using training set predictions
		SNR = estimate_SNR(y_train_pred[7:-7,:], y_train[7:-7, :])
	
		# Deconvolve	
		y_pred = deconvolve(y_pred, SNR)

	# Drop first and last 7 predictions to avoid convolution/smoothing artifacts
	y_test = y_test[7:-7, :]
	y_pred = y_pred[7:-7, :]
	y_train_pred = y_train_pred[7:-7, :]
	
	# Report metrics (as is)
	print_metrics(y_test, y_pred)
	
		
	# Setting those less than zero to zero.
	print('Inducing sparsity...')
	y_pred2 = y_pred.copy()
	y_pred2[y_pred < 0] = 0
	
	# Report metrics
	print_metrics(y_test, y_pred2)
	
	
	# Rescale y_pred so each feature has the same spread as the training targets
	expected_mean = y_train_pred.mean(axis=0)
	expected_std = y_train_pred.std(axis=0)
	desired_mean = y_train.mean(axis=0) 
	desired_std = y_train.std(axis=0)
	
	y_pred3 = (y_pred - expected_mean)/expected_std # normalize to unit std
	y_pred3 = (y_pred3 * desired_std) + desired_mean # give new spread
	y_pred3[y_pred3 < 0] = 0 # set any negative prediction to zero
	
	# Report metrics
	print('Inducing same spread + sparsity...')
	print_metrics(y_test, y_pred3)
	
def test_neural_network(model_filename, features_filename):
	""" Test the neural network models. """
	print('Testing model', model_filename[:-3])

	# Read features and targets
	X_test, y_test = read_inputs(features_filename, 'test_feats.h5')
	
	# Read model
	with h5py.File(model_filename, 'r') as model_file:
		weights_ih = np.array(model_file['weights_ih'])
		bias_ih = np.array(model_file['bias_ih'])
		weights_ho = np.array(model_file['weights_ho'])
		bias_ho = np.array(model_file['bias_ho'])
	
	# Make predictions
	hidden = np.dot(X_test, weights_ih.transpose()) + bias_ih
	y_pred = np.dot(hidden, weights_ho.transpose()) + bias_ho
	
	# Make predictions on training set (used to estimate SNR and re-scaling)
	X_train, y_train = read_inputs('train' + features_filename[4:],
								   'train_feats.h5')
	hidden = np.dot(X_train, weights_ih.transpose()) + bias_ih
	y_train_pred = np.dot(hidden, weights_ho.transpose()) + bias_ho
	
	# If the model predicts convolved features, deconvolve them
	if '_conv' in model_filename:
		# Estimate the signal-to-noise ratio using training set predictions
		SNR = estimate_SNR(y_train_pred[7:-7,:], y_train[7:-7, :])
		
		# Deconvolve
		y_pred = deconvolve(y_pred, SNR)

	# Drop first and last 7 predictions to avoid convolution/smoothing artifacts
	y_test = y_test[7:-7, :]
	y_pred = y_pred[7:-7, :]
	y_train_pred = y_train_pred[7:-7, :]
	
	# Report metrics (as is)
	print_metrics(y_test, y_pred)
	
		
	# Setting those less than zero to zero.
	print('Inducing sparsity...')
	y_pred2 = y_pred.copy()
	y_pred2[y_pred < 0] = 0
	
	# Report metrics
	print_metrics(y_test, y_pred2)
	
	
	# Rescale y_pred so each feature has the same spread as the training targets
	expected_mean = y_train_pred.mean(axis=0)
	expected_std = y_train_pred.std(axis=0)
	desired_mean = y_train.mean(axis=0) 
	desired_std = y_train.std(axis=0)
	
	y_pred3 = (y_pred - expected_mean)/expected_std # normalize to unit std
	y_pred3 = (y_pred3 * desired_std) + desired_mean # give new spread
	y_pred3[y_pred3 < 0] = 0 # set any negative prediction to zero
	
	# Report metrics
	print('Inducing same spread + sparsity...')
	print_metrics(y_test, y_pred3)
	
if __name__ == "__main__":
	main()
