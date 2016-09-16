# brain2sent
Mapping fMRI brain volumes to a joint low-dimensional space and using that to generate sentences describing what the subject is seeing.

# Results
You can see results [here](https://htmlpreview.github.io/?https://github.com/ecobost/brain2sent/blob/master/results.html) and check the numbers [here](https://github.com/ecobost/brain2sent/blob/master/results.md). They are average.

# Pipeline
A rough draft of what is needed to recreate the experiments. See the documentation in each file for details.

1. Process the BOLD activations
	1. Use generate_responses.py to extract BOLD activations and ROI info from the mat file. Saved with or without temporal smoothing at delays of 4, 5, 6 and 7 seconds.
	2. Use generate_wu_deconv.m to estimate voxel-specific HRFs and neural responses via Wiener deconvolution (as in Wu et al., 2013).
2. Create the image vectors from the experimental stimuli
	1. Use create_stim_images.py to put all stimuli images (from Stimuli.mat) into a single folder (/images) with names 'img_xx.png' for xx = 1 ... n
	2. Use img2vec.lua to create the feature vector representation of each image in the folder /images.
	3. Use generate_feats.py to generate the averaged and convolved feature representations for each second of stimuli.
3. Fit the models
	1. Use train_models.py to fit the different models.
4. Predict the sentences
	1. Use vec2seq.lua to create the sentences for each predicted image vector

Done!
