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
| f_4sec			| 0.116			| -			|
| f_5sec			| 0.116			| -			|
| f_6sec			| 0.102			| -			|
| f_7sec			| 0.084			| -			|
| f_4sec_smooth		| 0.127			| -			|
| f_5sec_smooth		| 0.127			| -			|
| f_6sec_smooth		| 0.115			| -			|
| f_7sec_smooth		| 0.097			| -			|
| f_conv			| 0.146			| -			|
| f_conv_smooth		| 0.178			| -			|
| f_deconv			| 0.121			| -			|
| roi_4sec			| 0.052			| -			|
| roi_5sec			| 0.054			| -			|
| roi_6sec			| 0.045			| -			|
| roi_7sec			| 0.033			| -			|
| roi_4sec_smooth	| 0.070			| -			|
| roi_5sec_smooth	| 0.071			| -			|
| roi_6sec_smooth	| 0.063			| -			|
| roi_7sec_smooth	| 0.048			| -			|
| roi_conv			| 0.074			| -			|
| roi_conv_smooth	| 0.101			| -			|
| roi_deconv		| 0.044			| -			|

\* Cross-validation R2 scores for nn models are those from scikit-learn's score_r2 metric which calculates a weighted average over all features (weight is the feature variance) rather than a simple arithmetic mean as defined below. Thus, nn R2 scores cannot be directly compared to the rest of models (which were calculated as below). Test scores however can be compared among all models. We apologize for the inconvenience :)

# Test results
All metrics are computed on the test set. For models that predict convolved features, they were deconvolved using Wiener deconvolution (signal-to-noise estimated using the training set). Because the expected features are positive, any prediction less than zero was set to zero.

Lower is better for RMSE, higher is better for r and R^2.

| Model				| RMSE	| r_feature	| r_sample	| R^2_feature	| R^2_sample	|
|:-----------------:|:-----:|:---------:|:---------:|:-------------:|:-------------:|
| nn_4sec			| 0.146 | 0.352 | 0.616 | 0.020 | 0.302 |
| nn_5sec			| 0.149 | 0.344 | 0.602 | -0.010 | 0.272 |
| nn_6sec			| 0.150 | 0.308 | 0.585 | -0.015 | 0.268 |
| nn_7sec			| 0.152 | 0.253 | 0.554 | -0.034 | 0.252 |
| nn_4sec_smooth	| 0.163 | 0.364 | 0.589 | -0.184 | 0.101 |
| nn_5sec_smooth	| 0.162 | 0.354 | 0.588 | -0.165 | 0.122 |
| nn_6sec_smooth	| 0.159	| 0.323 | 0.571 | -0.132 | 0.157 |
| nn_7sec_smooth	| 0.157 | 0.280 | 0.551 | -0.095 | 0.193 |
| nn_conv			| 0.149 | 0.343 | 0.586 | 0.016 | 0.289 |
| nn_conv_smooth	| 0.154 | 0.351 | 0.578 | -0.057 | 0.215 |
| nn_deconv			| 0.149 | 0.345 | 0.604 | -0.002 | 0.277 |
| l2_4sec			| 0.139 | 0.350 | 0.639 | 0.099 | 0.382 |
| l2_5sec			| 0.140 | 0.341 | 0.633 | 0.090 | 0.372 |
| l2_6sec			| 0.142 | 0.301 | 0.613 | 0.065 | 0.349 |
| l2_7sec			| 0.145 | 0.246 | 0.590 | 0.037 | 0.324 |
| l2_4sec_smooth	| 0.145 | 0.361 | 0.628 | 0.033 | 0.302 |
| l2_5sec_smooth	| 0.146 | 0.353 | 0.622 | 0.025 | 0.297 |
| l2_6sec_smooth	| 0.147 | 0.321 | 0.607 | 0.018 | 0.294 |
| l2_7sec_smooth	| 0.148 | 0.274 | 0.588 | 0.008 | 0.290 |
| l2_conv			| 0.144 | 0.342 | 0.628 | 0.061 | 0.348 |
| l2_conv_smooth	| 0.144 | 0.350 | 0.613 | 0.056 | 0.332 |
| l2_deconv			| 0.140 | 0.338 | 0.632 | 0.089 | 0.370 |
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
