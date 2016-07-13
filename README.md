# brain2vec
Mapping fMRI brain volumes to a lower dimensional embedding space and using that to generate sentences describing what the subject is seing.

# Pipeline
1. Use create_stim_images.py to put all stimuli images (from Stimuli.mat) into a single folder /images with names 'img_xx.png' for xx = 1 ... n
2. Use img2vec.lua to create the feature vector representation of each image in the folder /images. Saved as train_feats.h5
[YET TO DO]
3. Use ... to create the different feature representations train_feats_?.h5 for each timestep in the experiment (see data_archive).
4. USe .. to deconvolve the bold signal
5. Use ... (scikit-learn) to fit the models
6. Use vec2seq to create the sentences for each predicted image vector

Done!
