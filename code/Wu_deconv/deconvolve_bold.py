# Written by: Erick Cobos T. (a01184587@itesm.mx)
# Date: 18-August-2016
""" Deconvolves the training and test BOLD activations using the HRFs and SNRs 
estimated (previously) as in Wu et al., 2013.

Called from generate_wu_deconv.m		
"""
import h5py
import numpy as np
from numpy.fft import fft, ifft	

def main():
	# Read training and test BOLD
	with h5py.File('train_bold.h5', 'r') as train_bold_file:
		train_bold = np.array(train_bold_file['responses'])
	with h5py.File('test_bold.h5', 'r') as test_bold_file:
		test_bold = np.array(test_bold_file['responses'])
	
	# Read HRFs and SNRs from Octave file
	with h5py.File('wu_estimates.h5', 'r') as wu_deconv_file:
		HRFs = np.array(wu_deconv_file['HRFs']['value']).transpose()
		SNRs = np.array(wu_deconv_file['SNRs']['value']).transpose()
			
	# Estimate neural responses from BOLD data
	train_deconv = deconvolve(train_bold, HRFs, SNRs)
	test_deconv = deconvolve(test_bold, HRFs, SNRs)
		
	# Normalize to zero mean and unit variance
	train_mean = train_deconv.mean(axis=0)
	train_std = train_deconv.std(axis=0)
	train_deconv =  (train_deconv - train_mean) / train_std
	test_deconv =  (test_deconv - train_mean) / train_std
	
	# Save neural responses
	with h5py.File('train_deconv.h5', 'w') as train_deconv_file:
		train_deconv_file.create_dataset('responses', data=train_deconv)
	with h5py.File('test_deconv.h5', 'w') as test_deconv_file:
		test_deconv_file.create_dataset('responses', data=test_deconv)
	
	# Resave HRFs and SNRs in h5py format
	with h5py.File('wu_estimates.h5', 'w') as estimates_file:
		estimates_file.create_dataset('HRFs', data=HRFs)
		estimates_file.create_dataset('SNRs', data=SNRs)
			
def deconvolve(signal, HRFs, SNRs):
	"""Deconvolve signal using Wiener deconvolution."""
	H = fft(HRFs, len(signal), axis=0)
	wiener_filter = np.conj(H) / (H * np.conj(H) + 1 / SNRs**2)
	deconvolved = np.real(ifft(wiener_filter * fft(signal, axis=0), axis=0))

	return deconvolved
	
if __name__ == "__main__":
	main()
