Description of data files.

## Brain responses
| 	   Filename			| 	Size 	| 				Description				|
|:---------------------:|:---------:|:--------------------------------------|
| train_bold.h5			| 259 Mb	| Matrix (7200 x 8982) called 'responses'. BOLD activations during training for Subject 1 (timesteps x voxels). Valid voxels are extracted using the provided ROI masks (9137 voxels out of 73728; all regions are used). Voxels with nan values are dropped (155). 								|
| roi_info.h5			| 74 Kb		| Vector (8982) called 'rois'. ROI ID for each voxel in train_bold (ID assignments in section ROIs below).					|
| train_bold_4s_delay.h5| 259 Mb	| Matrix (7200 x 8982) called 'responses'. BOLD activations at time t-4 seconds. Obtained by shifting each row in train_bold 4 positions ahead and filling the first 4 rows with nans.						|
| train_bold_5s_delay.h5| 259 Mb	| Matrix (7200 x 8982) called 'responses'. Similar to train_bold_4s_delay but for 5 seconds.								|
| train_bold_6s_delay.h5| 259 Mb	| Matrix (7200 x 8982) called 'responses'. Similar to train_bold_4s_delay but for 6 seconds.								|
| train_bold_7s_delay.h5| 259 Mb	| Matrix (7200 x 8982) called 'responses'. Similar to train_bold_4s_delay but for 7 seconds.								|
| train_bold_8s_delay.h5| 259 Mb	| Matrix (7200 x 8982) called 'responses'. Similar to train_bold_4s_delay but for 8 seconds.								|
| train_pd_deconv.h5	| 259 Mb	| Matrix (7200 x 8982) called 'responses'. Neural response estimated by deconvolving the canonical (Glover) HRF from the BOLD activations in train_bold via polynomial division.							|
| train_wu_deconv.h5	| 518 Mb	| Matrix (7200 x 8982) called 'responses'. Neural response deconvolved by using spontaneous events to estimate a voxel-specific HRF and Wiener deconvolution as described in Wu, 2013.							|
| train_wu_hrfs.h5		| 2 Mb		| Matrix (32 x 8982) called 'HRFs'. Voxel-specific HRFs estimated for the Wu deconvolution.									|
| test_xxxxx.h5			| -			| Matrix (540 x 8982) called 'test_responses'. Similar to the train_xxxx files above but for BOLD activity during testing.	|

## Image vectors
| 	   Filename			| 	Size	| 				Description				|
|:---------------------:|:---------:|:--------------------------------------|
| train_full_feats.h5	| 332 Mb	| Matrix (108000 x 768) called 'feats'. Each row corresponds to the convnet-generated feature vector for each image in the original stimuli matrix, thus, one vector per video frame.							|
| train_feats.h5		| 22 Mb		| Matrix (7200 x 768) called 'feats'. Each row is the feature vector for the image the subject was seeing at that timepoint. Obtained by subsampling 1 every 15 rows of train_full_feats.							|
| train_1sec_feats.h5	| 22 Mb		| Matrix (7200 x 768) called 'feats'. Each row is the mean feature vector for images spanning one second of video. Obtained by averaging every 15 rows in train_full_feats.								|
| train_conv_feats.h5	| 44 Mb		| Matrix (7200 x 768) called 'feats'. Each row is obtained by convolving the feature vectors in train_full_feats with the canonical (Glover) HRF and subsampling the results (1 every 15 rows).					|
| test_full_feats.h5	| 25 Mb		| Matrix (8100 x 768) called 'test_feats'. Similar to train_full_feats but for test images.	|
| test_feats.h5			|	-		| Matrix (540 x 768) called 'test_feats'. Similar to train_feats but for test images.		|
| test_1sec_feats.h5	|	-		| Matrix (540 x 768) called 'test_feats'. Similar to train_1sec_feats but for test images.	|
| test_conv_feats.h5	|	-		| Matrix (540 x 768) called 'test_feats'. Similar to train_conv_feats but for test images.	|

## Lasso models
All files have a matrix (768 x 8982) called 'weights' and a vector (768) called 'bias', the weights and bias for an l1-regularized multiple linear regression model (Lasso). They were trained on different input and desired output.
| 	   Filename			| 	Size	| 				Description				|
|:---------------------:|:---------:|:--------------------------------------|
| model_a.h5			| -			| Trained on train_bold_4s_delay to predict train_feats.		|
| model_b.h5			| -			| Trained on train_bold_4s_delay to predict train_1sec_feats.	|
| model_c.h5			| -			| Trained on train_bold_5s_delay to predict train_feats.		|
| model_d.h5			| -			| Trained on train_bold_5s_delay to predict train_1sec_feats.	|
| model_e.h5			| -			| Trained on train_bold_6s_delay to predict train_feats.		|
| model_f.h5			| -			| Trained on train_bold_6s_delay to predict train_1sec_feats.	|
| model_g.h5			| -			| Trained on train_bold_7s_delay to predict train_feats.		|
| model_h.h5			| -			| Trained on train_bold_7s_delay to predict train_1sec_feats.	|
| model_i.h5			| -			| Trained on train_bold_8s_delay to predict train_feats.		|
| model_j.h5			| -			| Trained on train_bold_8s_delay to predict train_1sec_feats.	|
| model_k.h5			| -			| Trained on train_pd_deconv to predict train_feats.			|
| model_l.h5			| -			| Trained on train_pd_deconv to predict train_1sec_feats.		|
| model_m.h5			| -			| Trained on train_wu_deconv to predict train_feats.			|
| model_n.h5			| -			| Trained on train_wu_deconv to predict train_1sec_feats.		|
| model_o.h5			| -			| Trained on train_bold to predict train_conv_feats.			|

## Neural network models
All files have a matrix (400 x 8982) called 'weights_ih' and a vector (400) called 'bias_ih' that map input to hidden layer activations and a matrix (768 x 400) called 'weights_ho'and a vector (768) called 'bias_ho' that map hidden layer activations to output. They were trained on different input and desired output.
| 	   Filename			| 	Size	| 				Description				|
|:---------------------:|:---------:|:--------------------------------------|
| model_p.h5			| -			| Trained on train_bold_4s_delay to predict train_feats.		|
| model_q.h5			| -			| Trained on train_bold_4s_delay to predict train_1sec_feats.	|
| model_r.h5			| -			| Trained on train_bold_5s_delay to predict train_feats.		|
| model_s.h5			| -			| Trained on train_bold_5s_delay to predict train_1sec_feats.	|
| model_t.h5			| -			| Trained on train_bold_6s_delay to predict train_feats.		|
| model_u.h5			| -			| Trained on train_bold_6s_delay to predict train_1sec_feats.	|
| model_v.h5			| -			| Trained on train_bold_7s_delay to predict train_feats.		|
| model_w.h5			| -			| Trained on train_bold_7s_delay to predict train_1sec_feats.	|
| model_x.h5			| -			| Trained on train_bold_8s_delay to predict train_feats.		|
| model_y.h5			| -			| Trained on train_bold_8s_delay to predict train_1sec_feats.	|
| model_z.h5			| -			| Trained on train_pd_deconv to predict train_feats.			|
| model_aa.h5			| -			| Trained on train_pd_deconv to predict train_1sec_feats.		|
| model_ab.h5			| -			| Trained on train_wu_deconv to predict train_feats.			|
| model_ac.h5			| -			| Trained on train_wu_deconv to predict train_1sec_feats.		|
| model_ad.h5			| -			| Trained on train_bold to predict train_conv_feats.			|

## Image vector predictions from best model
| 	   Filename			| 	Size 	| 				Description				|
|:---------------------:|:---------:|:--------------------------------------|
| train_preds_x.h5		| -			| Matrix (7200 x 768) called 'preds'. Each row is the image vector predicted by model_x (the best model) for each training sample.	|
| test_preds_x.h5		| -			| Matrix (540 x 768) called 'test_preds'. Similar to train_preds_x but for test samples.											|

## Sentence predictions (used only for qualitative comparisons, not training/cv)
| 	   Filename			| 	Size 	| 				Description				|
|:---------------------:|:---------:|:--------------------------------------|
| train_sents_gt.txt	| -			| Text file (7200 lines). Each line is the ground truth sentence prediction made by neuralTalk for each image in the training set (generated from train_feats).												|
| train_sents_x.txt		| -			| Text file (7200 lines). Each line is the sentence prediction for image vectors predicted by model_x (the best model) in the training set (generated from train_preds_x).								|
| test_sents_gt.txt		| -			| Text file (540 lines). Similar to train_sents_gt but for the test set (generated from test_feats).	|
| test_sents_x.txt		| -			| Text file (540 lines). Similar to train_sents_x but for the test set (generated from test_preds_x).	|

## ROIs (Subject 1)
| ID	| Name		|	Number of voxels	|
|:-----:|:---------:|:---------------------:|
| 24	| IPlh		| 943					|
| 32	| IPrh		| 1308					|
| 40	| MTlh		| 165					|
| 48	| MTplh		| 58					|
| 56	| MTprh		| 91					|
| 64	| MTrh		| 152					|
| 72	| OBJlh		| 119					|
| 80	| OBJrh		| 44					|
| 104	| RSCrh		| 71					|
| 112	| STSrh		| 45					|
| 120	| VOlh		| 111					|
| 128	| VOrh		| 299					|
| 136	| latocclh	| 309					|
| 144	| latoccrh	| 413					|
| 152	| v1lh		| 490					|
| 160	| v1rh		| 504					|
| 168	| v2lh		| 715					|
| 176	| v2rh		| 762					|
| 184	| v3alh		|  92					|
| 192	| v3arh		| 160					|
| 200	| v3blh		| 104					|
| 208	| v3brh		| 152					|
| 216	| v3lh		| 581					|
| 224	| v3rh		| 560					|
| 232	| v4lh		| 308					|
| 240	| v4rh		| 426					|

Note: Original data provided by Gallant's Lab (http://crcns.org/data-sets/vc/vim-2/about-vim-2)
