#{
Written by: Erick Cobos T (a01184587@itesm.mx)
Date: 18-July-2016

Estimates a voxel-specific HRF and a signal-to-noise ratio as in Wu et al., 2013.
And deconvolves the BOLD data using Wiener deconvolution.

I use the canonical HRF plus first derivative ('p_m=2' in line 27) because it 
gives me more biologically feasible HRFs and less noisy neural responses than 
using the first and second derivative. Also, I added 'xBF.T = 16;' (line 151) to
avoid having to install spm. If unset, spm_get_bf.m will call 
spm_get_defaults('stats.fmri.t') in line 124, which returns the same result: 16.

Prerequisites are (need to be in the same folder):
	spm_get_bf.m*
	spm_hrf.m*
	spm_orth.m*
	spm_Gpdf.m*
	wgr_deconv_canonhrf_par.m
	resave_wu_deconv_in_h5py_format.py

* I got the spm_xxxxxx.m files from neurodebian (github.com/neurodebian/spm12).
#}

# Load BOLD data (transposed because it comes from python/h5py)
load train_bold.h5;
bold = responses';

# Estimate HRF and SNRs
[deconv_bold, spike_times, HRFs, event_delays, hrf_properties, lambda] = wgr_deconv_canonhrf_par(data=bold, thr=1.75, event_lag_max=10, TR=1);
SNRs = 1 ./ sqrt(lambda);

# Save estimated HRFs and SNRs
save -hdf5 wu_estimates.h5 HRFs SNRs

# Deconvolve training and test BOLD
system('python3 deconvolve_bold.py')
