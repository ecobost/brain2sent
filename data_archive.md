Description of data files.

## Image vectors
| 	   Filename		| ~Size | 			Description				|
|:-----------------:|:-----:|:----------------------------------|
| train_feats.h5	| -		| Matrix (108000 x 768) called 'feats' where each row correspond to the CNN feature vector of the image in that position in the original stimulus matrix.												|
| train_feats_a.h5	| -		| Matrix (7200 x 768) called 'feats' where each row is the feature vector from the image the subject was seeing at that timepoint (1 every 15 rows of train_feats).											|
| train_feats_b.h5	| -		| Matrix (7200 x 768) called 'feats' where each row is the average feature vector of all images from the current timestep to the previous timestep (average of every 15 rows in train_feats).				|
| train_feats_c.h5	| -		| Matrix (7200x768) called 'feats' where each row is obtained by convolving the feature vectors in train_feats with the HRF and subsampling the results (1 every 15 rows)									|
| test_feats.h5		|	-	| Matrix (8100 x 768) called 'test_feats'. Same as train_feats but for test images	|
| test_feats_a.h5	|	-	| Matrix (540 x 768) called 'test_feats'. Same as train_feats_a but for test images	|
| test_feats_b.h5	|	-	| Matrix (540 x 768) called 'test_feats'. Same as train_feats_b but for test images	|
| test_feats_c.h5	|	-	| Matrix (540 x 768) called 'test_feats'. Same as train_feats_c but for test images	|

## Brain volumes
| 	   Filename		| ~Size | 			Description				|
|:-----------------:|:-----:|:----------------------------------|
| train_bold.h5		| -		| Matrix (7200 x 73728) called 'volumes' where each row corresponds to the BOLD responses of a linearized brain volume (64 x64 x18) during training (7200 timesteps)										|
| train_deconv.h5 | -		| Matrix (7200 x73728) called 'volumes' where each row is the deconvolved neural response (obtained from train_bold) of a linearized brain volume during training.											|
| test_bold.h5		| -		| Matrix (540 x 73728) called 'test_volumes'. Same as train_bold.	
| test_deconv.h5	| -		| Matrix (540 x 73728) called 'test_volumes'. Same as train_deconv.

## Models
| 	   Filename		| ~Size | 			Description				|
|:-----------------:|:-----:|:----------------------------------|
| model_a.h5		| -		| Matrix (73728 x 768) called 'weights' with the weights for the l1-regularized multiple linear regression model trained on train_deconv to predict train_feats_a											|
| model_b.h5		| -		| Matrix (73728 x 768) called 'weights' with the weights for the l1-regularized multiple linear regression model trained on train_deconv to predict train_feats_b											|
| model_c.h5		| -		| Matrix (73728 x 768) called 'weights' with the weights for the l1-regularized multiple linear regression model trained on train_bold to predict train_feats_c											|


## Image vector predictions (from diff models)
| 	   Filename		| ~Size | 			Description				|
|:-----------------:|:-----:|:----------------------------------|
| train_preds_a.h5	| -		| Matrix (7200 x 768) called 'preds' where each row is the image vector predicted by model_a for each training sample.					|
| train_preds_b.h5	| -		| Matrix (7200 x 768) called 'preds' where each row is the image vector predicted by model_b for each training sample.					|
| train_preds_c.h5	| -		| Matrix (7200 x 768) called 'preds' where each row is the image vector predicted by model_c for each training sample.					|
| test_preds.h5		| -		| Matrix (540 x 768) called 'test_preds' where each row is the image vector predicted by model_x (the best model) for each test sample.	|

## Sentence predictions (used only for qualitative comparisons, not training/cv)
| 	   Filename		| ~Size | 			Description				|
|:-----------------:|:-----:|:----------------------------------|
| train_sents.txt	| -		| Text file (7200 lines) where each line is the ground truth sentence prediction for each image in the training set (generated from train_feats_a): what the neuralTalk model would predict on the images. |
| train_sents_a.txt	| -		| Text file (7200 lines) where each line is the predicted sentence for image vectors predicted by model_a (generated from train_preds_a).	|
| train_sents_a.txt	| -		| Text file (7200 lines) where each line is the predicted sentence for image vectors predicted by model_b (generated from train_preds_b).	|
| train_sents_a.txt	| -		| Text file (7200 lines) where each line is the predicted sentence for image vectors predicted by model_c (generated from train_preds_c).	|
| test_sents.txt	| -		| Text file (540 lines) where each line is the predicted sentence for image vectors predicted by model_x (the best model) (generated from train_preds_x).	|


Note: Original data provided by Gallant's Lab (http://crcns.org/data-sets/vc/vim-2/about-vim-2)
