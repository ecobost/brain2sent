# Written by: Erick Cobos T
# Date: 20-July-2016
""" Fits the many different models using l1-regularized linear regression. 

Models try to predict the feature representations (image vectors) of the image
the subject was seeing given the BOLD activations.
See data_archive.md for a description of each model.
"""

import h5py
import numpy as np
from sklearn import linear_model


#### Model a

# Read regression features
Xs_file = h5py.File('train_bold_4s_delay.h5', 'r')
Xs = np.array(Xs_file['responses'])
Xs_file.close()

# Read regression targets
y_file = h5py.File('train_feats.h5', 'r')
y = np.array(y_file['feats'])
y_file.close()

# Drop first and last 10 samples to avoid convolution/deconvolution artifacts
Xs = Xs[10:-10, :]
y = y[10:-10, :]

# Read ROI assignments
roi_file = h5py.File('roi_info.h5', 'r')
roi = np.array(roi_file['rois'])
roi_file.close()


# If i do feature selection, anyway give back weights for all voxels with zeros in other positions so that I can keep an standard weight matrix.







# Select the regularization parameter (called alpha here) via cv

# I may have better results selecting an alpha per each of the 768 features.
for each feature (768):
LASSOCV and select the best

# Or maybe just do feature selection somehow. Lasso for feature selection (dubbed L1 feature selection) works fine

# I could do a couple of things to preprocess the outputs, scale it so it has the same covariance as the original feature vector and threshold negative outputs to zero or threshold such that sparsity is the same.
# I can justify this by saying that feature vectors are all positive and sparse, so we can use that information to refine the predictions.
# this will improve corr but maybe using the negatives for sentence prediction is better. Save it until the end. Present everything as normal. and then say that there is two things one could do to improve results.

X_train = Xs[:-718, :]; y_train = y[:-718, 3]; X_test = Xs[-718:,:]; y_test = y[-718:,3]
Test baseline (predicting y_train.mean) = 0.025472971, 
selectKBest 10. LassoCV (7 features, alpha = 0.00128, 0.02305 cv mse): 0.02601 mse, -0.06 corrcoef. RidgeCV (alpha 1000, 0.02296 mse): 0.02609 mse -0.034 corrcoef
selectKBest 100. LassoCV (16 feats, alpha = 0.0031, 0.02302 mse): 0.02566 test mse, 0.0345 corr. RidgeCV (alpha 3333, cv mse 0.02268): 0.02583 test mse, 0.043 corr
selectKbest 300 LassoCV (39 feats, alpha = 0.0031, 0.02298 cv mse): 0.02551 test mse, 0.06 test corr. RidgeCV(alpha = 3333, cv mse 0.02214): mse 0.02610, 0.05 corr
selectKbest 1000 LassoCV (129 feats, alpha = 0.0026, 0.02299 cv mse): 0.0259 test mse, 0.062 test corr. RidgeCV(alpha = 10000, cv mse 0.02149): mse 0.02641, 0.0389 corr
All features. LassoCV (0 feats, alpha = 0.01073, 0.02331 cv mse): 0.0254 test mse, 0 test corr. RidgeCV(alpha = 33333, 0.02246 cv mse ): mse 0.02616,  corr 0.0347
# Rather than fitting all in a LassoCv, use selectFrom MOdle and use the Lasso to selecte the best features.
# For lasso, set eps to something higher than the default 0.001, that way te alpas produced won't be that small, maybe 0.1-0.01 is enough, that way i only get sparse models.

# Factor analysis, 10 features and ridge(alpha=31600) gave 0.025457.


# Testing different feature selections
num_features = [1, 3, 10, 31, 100, 316, 1000, 3160]
ridge_alphas = [10, 31.6, 100, 316, 1000, 3160, 10000, 31600, 100000, 316000, 1000000, 3160000]
X_new = X_train
X_test_new = X_test
selector = linear_model.LinearRegression(n_jobs=-1).fit(X_new, y_train);
selector = ExtraTreesRegressor(n_estimators=1000, max_features=1000, n_jobs = -1, verbose=2, random_state=123).fit(X_train, y_train);
masks = {}
masks['v1'] = (roi == 152) | (roi == 160)
masks['v2'] = (roi == 168) | (roi == 176)
masks['v3'] = (roi == 216) | (roi == 224)
masks['v3a'] = (roi == 184) | (roi == 192)
masks['v3b'] = (roi == 200) | (roi == 208)
masks['v4'] = (roi == 232) | (roi == 240)
masks['lo'] = (roi == 136) | (roi == 144)
masks['mt'] = (roi == 48) | (roi == 56) | (roi == 64) | (roi == 72) #v5
masks['ip'] = (roi == 24) | (roi == 32)
masks['vo'] = (roi == 120) | (roi == 128)
masks['OBJ'] = (roi == 72) | (roi == 80) #?
masks['rsc'] = (roi == 104)
masks['sts'] = (roi == 112)
for ROI_name, mask in masks.items():
	print(ROI_name,':', mask.sum(), 'features'); X_new = X_train[:, mask]; X_test_new = X_test[:, mask];
#for k in num_features:
#	print(k, 'feature(s)'); max_k = selector.feature_importances_.argsort()[-k:]; X_new = X_train[:, max_k]; X_test_new = X_test[:, max_k];

#	print(k, 'feature(s)'); max_k = selector.coef_.argsort()[-k:]; X_new = X_train[:, max_k]; X_test_new = X_test[:, max_k];

#	print(k, 'feature(s)'); selector = linear_model.LinearRegression(n_jobs=-1); selector.fit(X_new, y_train); max_k = selector.coef_.argsort()[-k:]; X_new = X_new[:, max_k]; X_test_new = X_test_new[:, max_k];
	
# print(k, 'feature(s)' selector = SelectKBest(f_regression, k=k); selector.fit(X_train, y_train); X_new = selector.transform(X_train); 

	# estimator = linear_model.LinearRegression(n_jobs=-1); selector = RFECV(estimator, cv=5, verbose=2,  step=500); selector.fit(X_train, y_train); X_new = selector.transform(X_train); X_test_new = selector.transform(X_test); print(X_train.shape[1], 'feature(s)'); # Takes too much time, no results.
	
	model = linear_model.RidgeCV(alphas=ridge_alphas, cv=5); model.fit(X_new, y_train); y_pred = model.predict(X_test_new); mse = ((y_test - y_pred)**2).mean(); print('Ridge: alpha =', model.alpha_, 'mse =', mse)
	
	model = linear_model.LassoCV(cv=5, n_jobs=-1, selection='random', random_state=123, eps = 0.005, n_alphas=50); model.fit(X_new, y_train); print('Max', model.alphas_.max(), 'Min', model.alphas_.min()); sparsity = (model.coef_ != 0).sum(); y_pred = model.predict(X_test_new); mse = ((y_test - y_pred)**2).mean(); print('Lasso: alpha=', model.alpha_, 'sparsity =', sparsity , 'mse =', mse)

# Results for selectKbest
1 feature(s)
Ridge: alpha = 316 mse = 0.0258117094039
Lasso: alpha= 0.000156283378306 sparsity = 1 mse = 0.0258266139439
3 feature(s)
Ridge: alpha = 1000 mse = 0.0258835706201
Lasso: alpha= 0.000102385704091 sparsity = 3 mse = 0.0259370260223
10 feature(s)
Ridge: alpha = 3160 mse = 0.0259983273539
Lasso: alpha= 0.00116964207687 sparsity = 10 mse = 0.0260271814606
31 feature(s)
Ridge: alpha = 10000 mse = 0.0258347768701
Lasso: alpha= 0.0018600569911 sparsity = 18 mse = 0.0258663613068
100 feature(s)
Ridge: alpha = 10000 mse = 0.0257579293776
Lasso: alpha= 0.00295800918808 sparsity = 35 mse = 0.0256816528089
316 feature(s)
Ridge: alpha = 10000 mse = 0.0258827728808
Lasso: alpha= 0.00295800918808 sparsity = 81 mse = 0.0255629871354
1000 feature(s)
Ridge: alpha = 10000 mse = 0.026413326726
Lasso: alpha= 0.00280940014191 sparsity = 237 mse = 0.0257957783639
3160 feature(s)
Ridge: alpha = 31600 mse = 0.0263577771123
Lasso: alpha= 0.0107311801207 sparsity = 1 mse = 0.0254729715598

# Results for RFE with LinearRegression to select best features
# Results from the last rows may be invalid as the featurs selected have been 
# overfitted to the cv set.
3160 feature(s)
Ridge: alpha = 1000000 mse = 0.0255331176342
Lasso: alpha= 0.00960345229021 sparsity = 2 mse = 0.0254849749367
1000 feature(s)
Ridge: alpha = 316000 mse = 0.0255775853121
Lasso: alpha= 0.00742160421065 sparsity = 4 mse = 0.0255243365531
316 feature(s)
Ridge: alpha = 31600 mse = 0.0257987033165
Lasso: alpha= 0.00635828563707 sparsity = 7 mse = 0.0255552390595
100 feature(s)
Ridge: alpha = 3160 mse = 0.0265557553368
Lasso: alpha= 0.000205548118662 sparsity = 100 mse = 0.027506425309
31 feature(s)
Ridge: alpha = 1000 mse = 0.0259694872105
Lasso: alpha= 1.06463127597e-05 sparsity = 31 mse = 0.0261994342557
10 feature(s)
Ridge: alpha = 1000 mse = 0.025884510614
Lasso: alpha= 0.000313752341445 sparsity = 10 mse = 0.0259455408479
3 feature(s)
Ridge: alpha = 1000 mse = 0.0257912352207
Lasso: alpha= 1.06463127597e-05 sparsity = 3 mse = 0.0258589538366
1 feature(s)
Ridge: alpha = 1000 mse = 0.0256539849329
Lasso: alpha= 3.2884794994e-05 sparsity = 1 mse = 0.0256959849414

# Select from model using linear regression to calculate importance of features
# Same as above but without refitting the model to each subset.
# up to (and including) 31 feats they all use the max sparsity in Lasso.
1 feature(s)
Ridge: alpha = 10000 mse = 0.0255099023401
Lasso: alpha= 0.00416613950474 sparsity = 0 mse = 0.0254729715591
3 feature(s)
Ridge: alpha = 100000 mse = 0.0254768920029
Lasso: alpha= 0.00416613950474 sparsity = 0 mse = 0.0254729715591
10 feature(s)
Ridge: alpha = 100000 mse = 0.0254871694291
Lasso: alpha= 0.00582108126694 sparsity = 1 mse = 0.0254729715597
31 feature(s)
Ridge: alpha = 3160000 mse = 0.0254738169545
Lasso: alpha= 0.00582108126694 sparsity = 0 mse = 0.0254729715591
100 feature(s)
Ridge: alpha = 100000 mse = 0.0255181758682
Lasso: alpha= 0.00559176400735 sparsity = 4 mse = 0.0255061750198
316 feature(s)
Ridge: alpha = 100000 mse = 0.0255204384047
Lasso: alpha= 0.00559176400735 sparsity = 12 mse = 0.0255089591325
1000 feature(s)
Ridge: alpha = 316000 mse = 0.025523377976
Lasso: alpha= 0.00937138831122 sparsity = 1 mse = 0.0254749066623
3160 feature(s)
Ridge: alpha = 1000000 mse = 0.0255331176342
Lasso: alpha= 0.00960345229021 sparsity = 2 mse = 0.0254849749367

# Select from model using L1 wouldn't work as most features would be set to zero.

# Select from model using as selector an ExtraTrees(num_estimators=1000, max_features=1000)
1 feature(s)
Ridge: alpha = 1000 mse = 0.0254816722975
Lasso: alpha= 0.000204100972254 sparsity = 1 mse = 0.0254884874834
3 feature(s)
Ridge: alpha = 1000 mse = 0.0253992125651
Lasso: alpha= 1.10799793522e-05 sparsity = 3 mse = 0.0253979276915
10 feature(s)
Ridge: alpha = 3160 mse = 0.0252625238491
Lasso: alpha= 9.15264146264e-05 sparsity = 10 mse = 0.0252324218039 # eps 0.01
Lasso: alpha= 9.15264146264e-11 sparsity = 10 mse = 0.0252307422334 # eps 1e-8
Lasso: alpha= 1e-15 sparsity = 10 mse = 0.0252307422318 # alpha=1e-15
31 feature(s)
Ridge: alpha = 3160 mse = 0.0254194042357
Lasso: alpha= 0.00119378636398 sparsity = 24 mse = 0.0254504561668
100 feature(s)
Ridge: alpha = 3160 mse = 0.0255690913878
Lasso: alpha= 0.00217155050002 sparsity = 53 mse = 0.0253834879475
316 feature(s)
Ridge: alpha = 10000 mse = 0.0257246007056
Lasso: alpha= 0.00262061116219 sparsity = 105 mse = 0.0254317610879
1000 feature(s)
Ridge: alpha = 10000 mse = 0.026090643438
Lasso: alpha= 0.00381652307619 sparsity = 124 mse = 0.0254273957106
3160 feature(s)
Ridge: alpha = 31600 mse = 0.0258623358772
Lasso: alpha= 0.00976857371159 sparsity = 2 mse = 0.0254950544306

# Select features from different anatomical areas
v3a : 252 features
Ridge: alpha = 316000 mse = 0.0255246688235
Lasso: alpha= 0.00817668495761 sparsity = 0 mse = 0.0254729715591
OBJ : 163 features
Ridge: alpha = 100000 mse = 0.0255065133214
Lasso: alpha= 0.00687755995889 sparsity = 0 mse = 0.0254729715591
vo : 410 features
Ridge: alpha = 1000000 mse = 0.0254928135526
Lasso: alpha= 0.00731607093885 sparsity = 0 mse = 0.0254729715591
v1 : 994 features
Ridge: alpha = 10000000 mse = 0.025475436405
Lasso: alpha= 0.007865371673 sparsity = 3 mse = 0.0254834623684
mt : 420 features
Ridge: alpha = 316000 mse = 0.0254935648613
Lasso: alpha= 0.00784481875868 sparsity = 0 mse = 0.0254729715591
v2 : 1477 features
Ridge: alpha = 3160000 mse = 0.0254756828693
Lasso: alpha= 0.00713362810631 sparsity = 6 mse = 0.0254913652101
v3b : 256 features
Ridge: alpha = 100000 mse = 0.0255041027001
Lasso: alpha= 0.00807775277228 sparsity = 0 mse = 0.0254729715591
lo : 722 features
Ridge: alpha = 316000 mse = 0.025525531883
Lasso: alpha= 0.00775832729617 sparsity = 1 mse = 0.0255489795529
ip : 2251 features
Ridge: alpha = 1000000 mse = 0.0255565696404
Lasso: alpha= 0.0094853239473 sparsity = 1 mse = 0.0254729715603 #smallest sparsity
v4 : 734 features
Ridge: alpha = 1000000 mse = 0.0254969086601
Lasso: alpha= 0.00903109875911 sparsity = 0 mse = 0.0254729715591
sts : 45 features
Ridge: alpha = 31600 mse = 0.0254427225328
Lasso: alpha= 0.00345015865989 sparsity = 6 mse = 0.025463378514






from scipy import ndimage
X_new = ndimage.gaussian_filter1d(X_train, 15, 1)
X_new = (X_new - X_new.mean(0))/X_new.std(0)
X_test_new = ndimage.gaussian_filter1d(X_test, 15, 1)
X_test_new = (X_test_new - X_new.mean(0))/X_new.std(0)



selector =  SelectKBest(f_regression, k=316).fit(X_train, y_train)
X_new = selector.transform(X_train)
model = linear_model.LassoCV(cv=5, n_jobs=-1, selection='random', random_state=123, verbose =2, eps = 0.5, n_alphas=50)
model.fit(X_new, y_train)
model.mse_path_.mean(axis=1)
model.mse_path_.mean(axis=1).min()
(model.coef_ > 0).sum()
model.alpha_
y_pred = model.predict(selector.transform(X_test))
((y_pred-y_test)**2).mean()
np.corrcoef(y_pred, y_test)

model = linear_model.RidgeCV(alphas = [0.0001, 0.001, 0.01], cv=5)
model.fit(X_new, y_train)
model.cv_values_.mean(axis=0)
y_pred = model.predict(X_test)
((y_pred-y_test)**2).mean()
np.corrcoef(y_pred, y_test)

# All I have to do is feature selection then Lasso, if that's not good enough, then it is not good enough.


# Check ridge_regression cv
small_X_F = np.array(small_X, order='F')

model = linear_model.LassoCV(alphas = [0.005], max_iter=10000, cv=5, n_jobs=-1, selection = 'random', random_state=123)
t1 = time.time()
model.fit(X_train, y_train)
t_lasso_cv = time.time() - t1
print(t_lasso_cv)

# TIMING. For lasso smallX
model = linear_model.LassoCV(alphas = [0.001], max_iter=10000, cv=5)
time = 57.28
model = linear_model.LassoCV(alphas = [0.001], max_iter=10000, cv=5, n_jobs=-1)
time = 43.35
Same as above but with X in fOrtran order. All others are with normal X, not Fortran
time =  42.68
model = linear_model.LassoCV(alphas = [0.001], max_iter=10000, cv=5, n_jobs=-1, selection = 'random', random_state=123)
time = 4.31, 4.02 (another run)
model = linear_model.LassoCV(alphas = [0.001], max_iter=10000, cv=5, n_jobs=-1, selection = 'random', random_state=0)
time = 3.98
model = linear_model.LassoCV(alphas = [0.001], max_iter=10000, cv=5, n_jobs=-1,  selection = 'random', random_state=123, precompute=True)
time = 4.01
model = linear_model.LassoCV(alphas = [0.001], max_iter=10000, cv=5, n_jobs=-1, precompute=True)
time = 43.72

# Maybe just fit 20 handpicked regularization parameters anc check that the best is not close to the side.


Results:
model1 = linear_model.LassoCV(alphas=[10, 1, 0.1, 0.01], cv=5)
time = 9.2401
best_alpha = 10
mean_mse  = [0.02355177,  0.02355177,  0.02355177,  0.02356373]


model = linear_model.LassoCV(n_alphas=4, cv=5)
time = 662.1341
best_alpha = 0.00907
alphas_ = [9.07485814e-03,   9.07485814e-04,   9.07485814e-05, 9.07485814e-06]

model2 = linear_model.LassoCV(n_alphas=2, cv=5)
time = 267
best_alpha = 0.0090748581409267007
alphas_ = [  9.07485814e-03,   9.07485814e-06]
mean_mse = [ 0.02357651,  0.09137521]

model1 = linear_model.LassoCV(alphas=[100, 0.05], cv=5)
mean_mse = array([ 0.02355177,  0.02355177])

model.LassoCV(alphas = [...], cv=5)
alphas_ = array([  1.00000000e+04,   3.16227766e+03,   1.00000000e+03,
         3.16227766e+02,   1.00000000e+02,   3.16227766e+01,
         1.00000000e+01,   3.16227766e+00,   1.00000000e+00,
         3.16227766e-01,   1.00000000e-01,   3.16227766e-02,
         1.00000000e-02,   3.16227766e-03,   1.00000000e-03,
         3.16227766e-04,   1.00000000e-04])
mean_mse = array([ 0.02355177,  0.02355177,  0.02355177,  0.02355177,  0.02355177,
        0.02355177,  0.02355177,  0.02355177,  0.02355177,  0.02355177,
        0.02355177,  0.02355177,  0.02356373,  0.02441403,  0.03039669,
        0.04486604,  0.06452146])
        
# ridge regression has the same complexity as ordinary regression, so ridge > ordinary plus ridge has the efficient CV, lasso seems to have the same efficiency. in secs.







































# Train model (search for best regularization parameter)
model = linear_model.LassoCV()
model.fit(Xs, y)


# DO k-fold crossvalidation to asses accuracy
cross-val-score (?)

# Train model in whole dataset

# print name of model, regularization params, accuracy
print("Model ", model_name)
print(model)
print("Alpha:", model.alpha)
print("Accuracy:"model.cvscores)

# Save model
model_file = h5py.File('model_a.h5', 'w')
model_file.create_dataset('weights', data=model.coef_)
model_file.create_dataset('bias', data=model.intercept_)
model_file.close()


# Save mse_path,mean and alphas tried

def train_model(features_filename, targets_filename, model_name):
	pass
	
if __name__ == "__main__":
	main()
