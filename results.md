# Results
All metrics are computed on the test set.
Lower is better for RMSE, higher is better for r and R^2.

| Model	| RMSE	| R^2_sample	| R^2_feature	| r_sample	| r_feature	| 
|:-----:|:-----:|:-------------:|:-------------:|:---------:|:---------:|
| a		| xxxx	| xxxx			| xxxxx			| xxx		| xxxx		|

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
