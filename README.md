# brain2sent
Mapping fMRI brain volumes to a joint low-dimensional space and using that to generate sentences describing what the subject is seeing.

# Pipeline
A rough draft of what is needed to recreate the experiments. See the documentation in each file for details.

1. Process the BOLD activations
	1. Use generate_responses.py to extract the BOLD activations at different time delays and estimate neural responses (deconvolved via polynomial division).
	2. Use generate_wu_deconv.m to estimate a voxel-specific HRF and estimate the neural responses via Wiener deconvolution (as in Wu et al.,2013).
2. Create the image vectors from the experimental stimuli
	1. Use create_stim_images.py to put all stimuli images (from Stimuli.mat) into a single folder (/images) with names 'img_xx.png' for xx = 1 ... n
	2. Use img2vec.lua to create the feature vector representation of each image in the folder /images. Saved as train_feats.h5
	3. Use generate_feats.py to generate the subsampled, averaged and deconvolved feature representations for each timestep.
3. Fit the models
	1. Use train_models.py to fit the different models. They are compared via cross validation.
4. Predict the sentences
	1. Use vec2seq.lua to create the sentences for each predicted image vector

Done!
