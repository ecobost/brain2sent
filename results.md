# Cross-validation results
Results computed using 5-fold cross-validation.
Alpha (the regularization parameter) was crossvalidated for neural networks. The R^2 is that of the best/selected configuration.

| Model				| R^2_feature*	| Best alpha|
|:-----------------:|:-------------:|:---------:|
| nn_4sec			| 0.131			| 112.88	|
| nn_5sec			| 0.134			| 106.25	|
| nn_6sec			| 0.114			| 112.88	|
| nn_7sec			| 0.086			| 127.43	|
| nn_4sec_smooth	| 0.142			| 65.43		|
| nn_5sec_smooth	| 0.143			| 73.86		|
| nn_6sec_smooth	| 0.124			| 83.38		|
| nn_7sec_smooth	| 0.098			| 94.12		|
| nn_conv			| 0.179			| 94.12		|
| nn_conv_smooth	| 0.198			| 83.38		|
| nn_deconv			| 0.125			| 106.25	|
| l2_4sec			| 0.080			| -			|
| l2_5sec			| 0.083			| -			|
| l2_6sec			| 0.069			| -			|
| l2_7sec			| 0.050			| -			|
| l2_4sec_smooth	| 0.089			| -			|
| l2_5sec_smooth	| 0.091			| -			|
| l2_6sec_smooth	| 0.079			| -			|
| l2_7sec_smooth	| 0.061			| -			|
| l2_conv			| 0.114			| -			|
| l2_conv_smooth	| 0.130			| -			|
| l2_deconv			| 0.074			| -			|
| f_4sec			| xxxxx			| -			|
| f_5sec			| xxxxx			| -			|
| f_6sec			| xxxxx			| -			|
| f_7sec			| 0.084			| -			|
| f_4sec_smooth		| xxxxx			| -			|
| f_5sec_smooth		| 0.127			| -			|
| f_6sec_smooth		| xxxxx			| -			|
| f_7sec_smooth		| xxxxx			| -			|
| f_conv			| xxxxx			| -			|
| f_conv_smooth		| xxxxx			| -			|
| f_deconv			| xxxxx			| -			|
| roi_4sec			| 0.052			| -			|
| roi_5sec			| xxxxx			| -			|
| roi_6sec			| xxxxx			| -			|
| roi_7sec			| xxxxx			| -			|
| roi_4sec_smooth	| xxxxx			| -			|
| roi_5sec_smooth	| xxxxx			| -			|
| roi_6sec_smooth	| xxxxx			| -			|
| roi_7sec_smooth	| xxxxx			| -			|
| roi_conv			| xxxxx			| -			|
| roi_conv_smooth	| xxxxx			| -			|
| roi_deconv		| 0.044			| -			|

\* Cross-validation R2 scores for nn models are those from scikit-learn's score_r2 metric which calculates a weighted average over all features (weight is the feature variance) rather than a simple arithmetic mean as defined below. Thus, nn R2 scores cannot be directly compared to the rest of models (which were calculated as below). Test scores however can be compared among all models. We apologize for the inconvenience :)

# Test results
All metrics are computed on the test set. For models that predict convolved features, they were deconvolved using Wiener deconvolution (signal-to-noise estimated using the training set). Because the expected features are positive, any prediction less than zero was set to zero.

Lower is better for RMSE, higher is better for r and R^2.

| Model				| RMSE	| R^2_sample	| R^2_feature	| r_sample	| r_feature	| 
|:-----------------:|:-----:|:-------------:|:-------------:|:---------:|:---------:|
| nn_4sec			| 
| nn_5sec			| 
| nn_6sec			| 
| nn_7sec			| 
| nn_4sec_smooth	| 
| nn_5sec_smooth	| 
| nn_6sec_smooth	| 
| nn_7sec_smooth	| 
| nn_conv			| 
| nn_conv_smooth	| 
| nn_deconv			| 
| l2_4sec			| 
| l2_5sec			| 
| l2_6sec			| 
| l2_7sec			| 
| l2_4sec_smooth	| 
| l2_5sec_smooth	| 
| l2_6sec_smooth	| 
| l2_7sec_smooth	| 
| l2_conv			| 
| l2_conv_smooth	| 
| l2_deconv			| 
| f_4sec			| 
| f_5sec			| 
| f_6sec			| 
| f_7sec			| 
| f_4sec_smooth		| 
| f_5sec_smooth		| 
| f_6sec_smooth		| 
| f_7sec_smooth		| 
| f_conv			| 
| f_conv_smooth		| 
| f_deconv			| 
| roi_4sec			| 
| roi_5sec			| 
| roi_6sec			| 
| roi_7sec			| 
| roi_4sec_smooth	| 
| roi_5sec_smooth	| 
| roi_6sec_smooth	| 
| roi_7sec_smooth	| 
| roi_conv			| 
| roi_conv_smooth	| 
| roi_deconv		| 

# Definitions
Assuming y_true is the target matrix (samples x feature units, i.e., 540 x 768),
y_pred is a matrix (540 x 768) with the predicted output.

|	Metric		|				Name			|		 	Definition		|
|:-------------:|:------------------------------|:--------------------------|
| MSE			| Mean squared error			| `[(y_true - y_pred)**2].mean()`		|
| RMSE			| Root mean squared error 		| `sqrt(MSE)`							|
| r_feature		| Mean output-wise correlation	| `corr_per_col(y_true, y_pred).mean()`	|
| r_sample		| Mean sample-wise correlation	| `corr_per_row(y_true, y_pred).mean()`	|
| RSS_feature	| Feature-wise residual sum of squares	| `[(y_true - y_pred)**2].sum(axis=0)`					|
| RSS_sample	| Sample-wise residual sum of squares	| `[(y_true - y_pred)**2].sum(axis=1)`					|
| TSS_feature	| Feature-wise total sum of squares		| `{[y_true - y_true.mean(axis=0)]**2}.sum(axis=0)`		|
| TSS_sample	| Sample-wise total sum of squares		| `{[y_true - y_true.mean(axis=1)]**2}.sum(axis=1)`		|
| R^2_feature	| Mean feature-wise coefficient of determination	| `1 - (RSS_feature/TSS_feature).mean()`	|
| R^2_sample	| Mean sample-wise coefficient of determination		| `1 - (RSS_sample/TSS_sample).mean()`		|
