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
#from sklearn.neural_network import MLPRegressor


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

X_train = Xs[:-718, :]; y_train = y[:-718, 3]; X_test = Xs[-718:,:]; y_test = y[-718:, 3]
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
selector = ExtraTreesRegressor(n_estimators=1000, max_features=300, n_jobs=-1, verbose=1, random_state=123).fit(X_train, y_train);
selector = ExtraTreesRegressor(n_estimators=1000, max_features=449, n_jobs=-1, verbose=1, random_state=123).fit(X_train, y_train);
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
	
# print(k, 'feature(s)'); selector = SelectKBest(f_regression, k=k); selector.fit(X_train, y_train); X_new = selector.transform(X_train); X_test_new = selector.transform(X_test);

	# estimator = linear_model.LinearRegression(n_jobs=-1); selector = RFECV(estimator, cv=5, verbose=2,  step=500); selector.fit(X_train, y_train); X_new = selector.transform(X_train); X_test_new = selector.transform(X_test); print(X_train.shape[1], 'feature(s)'); # Takes too much time, no results.
	
	model = linear_model.RidgeCV(alphas=ridge_alphas, cv=5); model.fit(X_new, y_train); y_pred = model.predict(X_test_new); mse = ((y_test - y_pred)**2).mean(); print('Ridge: alpha =', model.alpha_, 'mse =', mse)
	
	model = linear_model.LassoCV(cv=5, n_jobs=-1, selection='random', random_state=123, eps = 0.01, n_alphas=50); model.fit(X_new, y_train); print('Max', model.alphas_.max(), 'Min', model.alphas_.min()); sparsity = (model.coef_ != 0).sum(); y_pred = model.predict(X_test_new); mse = ((y_test - y_pred)**2).mean(); print('Lasso: alpha=', model.alpha_, 'sparsity =', sparsity , 'mse =', mse)

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

# ExtraTrees Regressor (1000 trees, 50 features)
1 feature(s)
Ridge: alpha = 3160000 mse = 0.0254730943734
Lasso: alpha= 0.00324058960769 sparsity = 0 mse = 0.0254729715591
3 feature(s)
Ridge: alpha = 3160 mse = 0.0255598109899
Lasso: alpha= 0.00203462897773 sparsity = 3 mse = 0.0255601943008
10 feature(s)
Ridge: alpha = 10000 mse = 0.0254512983129
Lasso: alpha= 0.00325511891118 sparsity = 5 mse = 0.025475881595
31 feature(s)
Ridge: alpha = 3160 mse = 0.0255118689784
Lasso: alpha= 0.00217155050002 sparsity = 20 mse = 0.0255375864498
100 feature(s)
Ridge: alpha = 10000 mse = 0.0256647528
Lasso: alpha= 0.00179943962773 sparsity = 58 mse = 0.0257634974134
316 feature(s)
Ridge: alpha = 10000 mse = 0.0258631041355
Lasso: alpha= 0.00287884917882 sparsity = 87 mse = 0.025478388486
1000 feature(s)
Ridge: alpha = 31600 mse = 0.0257268302126
Lasso: alpha= 0.00736855243542 sparsity = 11 mse = 0.0255465252096
3160 feature(s)
Ridge: alpha = 100000 mse = 0.0255759286749
Lasso: alpha= 0.00889231485123 sparsity = 4 mse = 0.0255173894363

# ExtraTrees (1000 trees, 316 features)
1 feature(s)
Ridge: alpha = 1000 mse = 0.0254816722975
Lasso: alpha= 0.000204100972254 sparsity = 1 mse = 0.0254884874834
3 feature(s)
Ridge: alpha = 3160 mse = 0.0253901593462
Lasso: alpha= 0.00121718131708 sparsity = 3 mse = 0.0253926954983
10 feature(s)
Ridge: alpha = 3160 mse = 0.0253268672669
Lasso: alpha= 0.00199678988039 sparsity = 8 mse = 0.0253379688386
31 feature(s)
Ridge: alpha = 3160 mse = 0.0251822438826
Lasso: alpha= 0.00230484062172 sparsity = 19 mse = 0.0252116175905
100 feature(s)
Ridge: alpha = 10000 mse = 0.0254866850837
Lasso: alpha= 0.00262061116219 sparsity = 41 mse = 0.0253618705487
316 feature(s)
Ridge: alpha = 10000 mse = 0.025865781306
Lasso: alpha= 0.00262061116219 sparsity = 106 mse = 0.0255458151504
1000 feature(s)
Ridge: alpha = 31600 mse = 0.0257292891518
Lasso: alpha= 0.00316253426451 sparsity = 175 mse = 0.0255058537775
3160 feature(s)
Ridge: alpha = 31600 mse = 0.025908114711
Lasso: alpha= 0.00976857371159 sparsity = 2 mse = 0.0254950185544

# ExtraTrees (1000 trees, 3160 feats)
1 feature(s)
Ridge: alpha = 1000 mse = 0.0254816722975
Lasso: alpha= 0.000204100972254 sparsity = 1 mse = 0.0254884874834
3 feature(s)
Ridge: alpha = 1000 mse = 0.0254520076518
Lasso: alpha= 0.00019412303037 sparsity = 3 mse = 0.0254648324569
10 feature(s)
Ridge: alpha = 3160 mse = 0.0252822580813
Lasso: alpha= 9.43812657825e-05 sparsity = 10 mse = 0.0252383605367 #eps 0.01
Lasso: alpha= 9.43812657825e-09 sparsity = 10 mse = 0.0252354255997 # eps 1e-6
31 feature(s)
Ridge: alpha = 3160 mse = 0.025383552543
Lasso: alpha= 0.000904990466984 sparsity = 27 mse = 0.0254750235523
100 feature(s)
Ridge: alpha = 10000 mse = 0.0254443308338
Lasso: alpha= 0.00217155050002 sparsity = 52 mse = 0.0253117897075
316 feature(s)
Ridge: alpha = 10000 mse = 0.0258294935812
Lasso: alpha= 0.00287884917882 sparsity = 97 mse = 0.0255179601548
1000 feature(s)
Ridge: alpha = 10000 mse = 0.0262805191784
Lasso: alpha= 0.00347417400251 sparsity = 151 mse = 0.0254408545368
3160 feature(s)
Ridge: alpha = 31600 mse = 0.0258966213606
Lasso: alpha= 0.00976857371159 sparsity = 2 mse = 0.0254963217239

# ExtraTrees (100 trees, 50 feats)
1 feature(s)
Ridge: alpha = 10000 mse = 0.0254660345062
Lasso: alpha= 2.33171355088e-05 sparsity = 1 mse = 0.025459159123 # eps 0.01
Lasso: alpha= 4.71843140303e-06 sparsity = 1 mse = 0.0254590987406 # eps 1e-6
3 feature(s)
Ridge: alpha = 3160 mse = 0.0254022470426
Lasso: alpha= 7.25881088304e-05 sparsity = 3 mse = 0.025394140475 # eps 0.01
Lasso: alpha= 0.00010571358178 sparsity = 3 mse = 0.0253944823272 # eps 1e-6
10 feature(s)
Ridge: alpha = 3160 mse = 0.0255468898675
Lasso: alpha= 0.00147930036114 sparsity = 7 mse = 0.0255476093773
31 feature(s)
Ridge: alpha = 10000 mse = 0.0255341612241
Lasso: alpha= 0.00313752341445 sparsity = 11 mse = 0.0255337837751
100 feature(s)
Ridge: alpha = 10000 mse = 0.0254373805752
Lasso: alpha= 0.00347417400251 sparsity = 26 mse = 0.0254908477359
316 feature(s)
Ridge: alpha = 31600 mse = 0.0255797316663
Lasso: alpha= 0.0067075798564 sparsity = 6 mse = 0.0255856360688
1000 feature(s)
Ridge: alpha = 31600 mse = 0.0254956834373
Lasso: alpha= 0.00736855243542 sparsity = 8 mse = 0.0255581097291
3160 feature(s)
Ridge: alpha = 316000 mse = 0.0255294425666
Lasso: alpha= 0.00889231485123 sparsity = 4 mse = 0.0255169266901

# ExtraTrees (100 trees, 316 feats)
1 feature(s)
Ridge: alpha = 1000 mse = 0.0254816722975
Lasso: alpha= 0.000204100972254 sparsity = 1 mse = 0.0254884874834
3 feature(s)
Ridge: alpha = 3160 mse = 0.0254471419647
Lasso: alpha= 8.17668495761e-05 sparsity = 3 mse = 0.0254853213874 # eps 0.01
Lasso: alpha= 8.17668495761e-09 sparsity = 3 mse = 0.0254871363796 # eps 1e-6
10 feature(s)
Ridge: alpha = 3160 mse = 0.0256619318916
Lasso: alpha= 0.00287884917882 sparsity = 6 mse = 0.025653122142
31 feature(s)
Ridge: alpha = 3160 mse = 0.0256479327297
Lasso: alpha= 0.0014910926427 sparsity = 22 mse = 0.025668267449
100 feature(s)
Ridge: alpha = 10000 mse = 0.0256424101185
Lasso: alpha= 0.00197675846358 sparsity = 45 mse = 0.0257474202316
316 feature(s)
Ridge: alpha = 10000 mse = 0.0259703938196
Lasso: alpha= 0.00347417400251 sparsity = 61 mse = 0.0255717905082
1000 feature(s)
Ridge: alpha = 31600 mse = 0.0255197618809
Lasso: alpha= 0.00736855243542 sparsity = 10 mse = 0.025543328824
3160 feature(s)
Ridge: alpha = 100000 mse = 0.0256205319715
Lasso: alpha= 0.00976857371159 sparsity = 2 mse = 0.0254949366458

# ExtraTrees (100 trees, 1000 feats)
1 feature(s)
Ridge: alpha = 1000 mse = 0.0254816722975
Lasso: alpha= 0.000204100972254 sparsity = 1 mse = 0.0254884874834
3 feature(s)
Ridge: alpha = 3160 mse = 0.0254799823869
Lasso: alpha= 0.000130815407269 sparsity = 3 mse = 0.0255237727003
10 feature(s)
Ridge: alpha = 10000 mse = 0.0253081246254
Lasso: alpha= 0.00817668495761 sparsity = 1 mse = 0.0254729715584 # max sparsity
31 feature(s)
Ridge: alpha = 10000 mse = 0.0253139599265
Lasso: alpha= 0.00915264146264 sparsity = 0 mse = 0.0254729715591
100 feature(s)
Ridge: alpha = 10000 mse = 0.0253964585115
Lasso: alpha= 0.00253196224868 sparsity = 32 mse = 0.0253159968324
316 feature(s)
Ridge: alpha = 10000 mse = 0.0254858465346
Lasso: alpha= 0.00319443410473 sparsity = 73 mse = 0.0251419850854
1000 feature(s)
Ridge: alpha = 31600 mse = 0.0255063829744
Lasso: alpha= 0.00378634016639 sparsity = 106 mse = 0.0253059381339
3160 feature(s)
Ridge: alpha = 100000 mse = 0.0255685943478
Lasso: alpha= 0.00889231485123 sparsity = 4 mse = 0.0255173720296

# ExtraTrees (100 trees, 3160 feats)
1 feature(s)
Ridge: alpha = 1000 mse = 0.0254816722975
Lasso: alpha= 0.000204100972254 sparsity = 1 mse = 0.0254884874834
3 feature(s)
Ridge: alpha = 1000 mse = 0.0254550201011
Lasso: alpha= 9.15264146264e-05 sparsity = 3 mse = 0.0254686079926 # eps 0.01
Lasso: alpha= 0.000100545532046 sparsity = 3 mse = 0.0254685600356 # eps 1e-6
10 feature(s)
Ridge: alpha = 3160 mse = 0.0253711319058
Lasso: alpha= 0.00168598059783 sparsity = 9 mse = 0.0253900095379
31 feature(s)
Ridge: alpha = 3160 mse = 0.0255248186596
Lasso: alpha= 0.000900487387353 sparsity = 28 mse = 0.0255467862799
100 feature(s)
Ridge: alpha = 10000 mse = 0.0255518650921
Lasso: alpha= 0.00238553756617 sparsity = 44 mse = 0.0254045673916
316 feature(s)
Ridge: alpha = 10000 mse = 0.0257365974084
Lasso: alpha= 0.00262061116219 sparsity = 91 mse = 0.0253371954802
1000 feature(s)
Ridge: alpha = 31600 mse = 0.0256881626702
Lasso: alpha= 0.00347417400251 sparsity = 128 mse = 0.0254443785842
3160 feature(s)
Ridge: alpha = 100000 mse = 0.0256017574467
Lasso: alpha= 0.00976857371159 sparsity = 2 mse = 0.0254964009497

# ExtraTrees (500 trees, 898 feats)
1 feature(s)
Ridge: alpha = 1000 mse = 0.0254816722975
Lasso: alpha= 0.000204100972254 sparsity = 1 mse = 0.0254884874834
3 feature(s)
Ridge: alpha = 3160 mse = 0.0254124771418
Lasso: alpha= 0.00110799793522 sparsity = 3 mse = 0.0254227774135
10 feature(s)
Ridge: alpha = 3160 mse = 0.025382435822
Lasso: alpha= 0.00245537527675 sparsity = 5 mse = 0.0254224174859
31 feature(s)
Ridge: alpha = 3160 mse = 0.0254393424954
Lasso: alpha= 0.0010921358354 sparsity = 26 mse = 0.0254892501748
100 feature(s)
Ridge: alpha = 10000 mse = 0.0254647322869
Lasso: alpha= 0.00262061116219 sparsity = 45 mse = 0.0252917497987
316 feature(s)
Ridge: alpha = 10000 mse = 0.0257317230582
Lasso: alpha= 0.00217155050002 sparsity = 123 mse = 0.0254398381885
1000 feature(s)
Ridge: alpha = 31600 mse = 0.0254975511697
Lasso: alpha= 0.00381652307619 sparsity = 123 mse = 0.0252583584086
3160 feature(s)
Ridge: alpha = 31600 mse = 0.0258410676056
Lasso: alpha= 0.00976857371159 sparsity = 2 mse = 0.0254951504389

# ExtraTrees (1000 trees, 449 features)
1 feature(s)
Ridge: alpha = 1000 mse = 0.0254816722975
Lasso: alpha= 0.000204100972254 sparsity = 1 mse = 0.0254884874834
3 feature(s)
Ridge: alpha = 3160 mse = 0.0254471419647
Lasso: alpha= 8.17668495761e-05 sparsity = 3 mse = 0.025485321106
10 feature(s)
Ridge: alpha = 10000 mse = 0.0254433138152
Lasso: alpha= 0.00357588204599 sparsity = 5 mse = 0.0254693652351
31 feature(s)
Ridge: alpha = 3160 mse = 0.0253559370571
Lasso: alpha= 0.00199669762435 sparsity = 23 mse = 0.0253403264678
100 feature(s)
Ridge: alpha = 10000 mse = 0.025435490171
Lasso: alpha= 0.00238553756617 sparsity = 47 mse = 0.0253860720153
316 feature(s)
Ridge: alpha = 10000 mse = 0.0258691998639
Lasso: alpha= 0.00262061116219 sparsity = 101 mse = 0.0254984261673
1000 feature(s)
Ridge: alpha = 10000 mse = 0.0265369355582
Lasso: alpha= 0.00316253426451 sparsity = 174 mse = 0.0256017172743
3160 feature(s)
Ridge: alpha = 31600 mse = 0.0260560028208
Lasso: alpha= 0.00976857371159 sparsity = 2 mse = 0.0254949951054


# Neural netowkrs(with 500 hidden units). Less time, less hassle :))), tol = 1e-8
# Baseline correlation: 0.526314
# alpha = 1
((y_pred[:,3] - y_test[:,3])**2).mean() = 0.0336
corr = 0.3723
#alpha = 10
((y_pred[:,3] - y_test[:,3])**2).mean() = 0.027
corr = 0.40725 (0.41103 adter setting y_pred < 0)
#alpha= 50
((y_pred[:,3] - y_test[:,3])**2).mean() = 0.026625
corr = 0.478427
corr/f = -0.00211
#alpha = 100
((y_pred[:,3] - y_test[:,3])**2).mean() = 0.02538
corr = 0.51424 (0.51424 after setting y_pred < 0)
corr/f = 0.01908
# alpha = 316
((y_pred[:,3] - y_test[:,3])**2).mean() = 0.0254806
corr = 0.526279
corr/f = 0.00615
# alpha= 500
((y_pred[:,3] - y_test[:,3])**2).mean() = 0.02545
corr = 0.52623
corr/f = 0.007605
#alpha= 1000
((y_pred[:,3] - y_test[:,3])**2).mean() = 0.0254739
corr = 0.52649 (0.52649 after setting y_pred < 0)
corr/f = -0.017349
# alpha = 10000
((y_pred[:,3] - y_test[:,3])**2).mean() = 0.0254723
corr = 0.526255

# Neural networks(600 hidden units)
#alpha = 100
((y_pred[:,3] - y_test[:,3])**2).mean() = 0.02538
corr = 0.5160
corr/f = 0.01390

#alpha = 150
((y_pred[:,3] - y_test[:,3])**2).mean() = 0.025296
corr = 0.528340
corr/f = 0.035078

#alpha = 300
((y_pred[:,3] - y_test[:,3])**2).mean() = 0.0254837
corr = 0.52625
corr/f = 0.00762

# Neural network (1000 hidden units)
#alpha=150
((y_pred[:,3] - y_test[:,3])**2).mean() = 0.0253706
corr = 0.5281075
corr/f = 0.032144

# Neural network 800->400
#alpha = 150
((y_pred[:,3] - y_test[:,3])**2).mean() = 0.026240
corr = 0.524087
corr/f = 0.00779

# Neural network 600 -> 600
# alpha = 500
((y_pred[:,3] - y_test[:,3])**2).mean() = 0.02556885
corr = 0.52446147
corr/f = -0.00107

# Neural network 500 -> 500
# alpha = 100
((y_pred[:,3] - y_test[:,3])**2).mean() = 0.025382
corr = 0.52358
corr/f = 0.003823202

# alpha = 500
((y_pred[:,3] - y_test[:,3])**2).mean() = 0.025755
corr = 0.527575
corr/f = 0.0023723



# Testing neural networks
hidden_layer_size = [50, 100, 300, 600, 1000, 1500, 2000]
regularization_params = [10, 21.5, 46.3, 100, 215, 463, 1000, 2150, 4630]

for layer_size in hidden_layer_size:
	print('Layer size:', layer_size)
	for alpha in regularization_params:
		print('alpha = ', alpha); model = MLPRegressor(hidden_layer_sizes = (layer_size,), alpha=alpha, max_iter=1000, random_state=123, early_stopping=True, tol = 1e-8); model.fit(X_train, y_train); y_pred = model.predict(X_test); mse = ((y_pred-y_test)**2).mean(); mse3 = ((y_pred[:,3] - y_test[:,3])**2).mean(); corr = 0; corrf = 0;
		for i in range(718):
			corr +=  np.corrcoef(y_pred[i,:], y_test[i,:])[0,1]
		for i in range(768):
			corrf +=  np.corrcoef(y_pred[:,i], y_test[:,i])[0,1]
		corr =	corr/718; corrf =	corrf/768; print('mse =', mse, 'mse3 =',mse3, 'corr =', corr, 'corr/f =', corrf)


# RESULTS (7:06pm-11:06pm)
# configs: MLPRegressor(activation='relu', algorithm='adam', alpha=10, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-08, hidden_layer_sizes=(100,), learning_rate='constant', learning_rate_init=0.001, max_iter=1000, momentum=0.9, nesterovs_momentum=True, power_t=0.5, random_state=123, shuffle=True, tol=1e-08, validation_fraction=0.1, verbose=False, warm_start=False)
# 50 hidden units
alpha =  10
mse = 0.0326135771442 mse3 = 0.028376172148 corr = 0.412102643693 corr/f = 0.00929452006918
alpha =  21.5
mse = 0.0304110586311 mse3 = 0.0273066888063 corr = 0.446544064574 corr/f = 0.0036036970825
alpha =  46.3
mse = 0.0285793978659 mse3 = 0.0262857848273 corr = 0.48050689623 corr/f = 0.0161413661382
alpha =  100
mse = 0.0264866791559 mse3 = 0.0255301851805 corr = 0.517756420731 corr/f = 0.0258963437105
alpha =  215
mse = 0.0262700833162 mse3 = 0.0254034716099 corr = 0.523628987347 corr/f = 0.0074048129554
alpha =  463
mse = 0.0260256696446 mse3 = 0.0254531440977 corr = 0.526473109528 corr/f = 0.0138420266665
alpha =  1000
mse = 0.026034969783 mse3 = 0.0254703088523 corr = 0.526218287638 corr/f =  0.00379268747948
alpha =  2150
mse = 0.026034949861 mse3 = 0.0254730920757 corr = 0.526215208047 corr/f =  0.00389357993785
alpha =  4630
mse = 0.0260352124609 mse3 = 0.0254735244087 corr =  0.526207047205 corr/f =  0.00371067046394
# 100 hidden units
alpha =  10
mse = 0.0331156551037 mse3 = 0.0294399899561 corr =  0.401688293754 corr/f =  0.00603382353788
alpha =  21.5
mse = 0.0313837626648 mse3 = 0.0277349678267 corr =  0.438905441585 corr/f =  -0.00297568629478
alpha =  46.3
mse = 0.0287721734822 mse3 = 0.0265572414357 corr =  0.481647255646 corr/f =  0.0143105467818
alpha =  100
mse = 0.0267376839654 mse3 = 0.0253431998704 corr =  0.514915081228 corr/f =  0.0117882372104
alpha =  215
mse = 0.0262577164125 mse3 = 0.0254680351977 corr =  0.523547153321 corr/f =  0.00838457625545
alpha =  463
mse = 0.0260220646099 mse3 = 0.0254658500222 corr =  0.52653111557 corr/f =  -0.00271227286095
alpha =  1000
mse = 0.0260346879257 mse3 = 0.0254678217472 corr =  0.526228807833 corr/f =  0.00748845919051
alpha =  2150
mse = 0.0260341820329 mse3 = 0.0254687470425 corr =  0.52624135492 corr/f =  -0.00315629830816
alpha =  4630
mse = 0.0260345705506 mse3 = 0.0254679642749 corr =  0.526230452443 corr/f =  0.000597529870995
# 300 hidden units
alpha =  10
mse = 0.032869706592 mse3 = 0.0289238012464 corr =  0.406288566289 corr/f =  0.000964269492139
alpha =  21.5
mse = 0.0311479811099 mse3 = 0.0280674977956 corr =  0.4367371886 corr/f =  -0.00242929982883
alpha =  46.3
mse = 0.0289962535585 mse3 = 0.0264576811979 corr =  0.475626275846 corr/f =  0.00529700357755
alpha =  100
mse = 0.0267612811674 mse3 = 0.0254863516365 corr =  0.515049775327 corr/f =  0.0111548053647
alpha =  215
mse = 0.0262399677402 mse3 = 0.0255059669612 corr =  0.524000751349 corr/f =  0.00723524473199
alpha =  463
mse = 0.0260359803233 mse3 = 0.0254509347628 corr =  0.52627397238 corr/f =  0.00752562702308
alpha =  1000
mse = 0.0260239579051 mse3 = 0.0254735556711 corr =  0.526489283672 corr/f =  -0.0188038257271
alpha =  2150
mse = 0.0260336425694 mse3 = 0.0254722150457 corr =  0.526252272917 corr/f =  0.00540687842707
alpha =  4630
mse = 0.0260340735495 mse3 = 0.0254726748552 corr =  0.526242005242 corr/f =  -0.000563484723196
# 600 hidden units
alpha =  10
mse = 0.0327665638029 mse3 = 0.0305435624021 corr =  0.410525688805 corr/f =  0.00601048503257
alpha =  21.5
mse = 0.0315467210406 mse3 = 0.0288479686131 corr =  0.42426028336 corr/f =  -0.00242101934168
alpha =  46.3
mse = 0.0294962446631 mse3 = 0.0264639898107 corr =  0.471442438371 corr/f =  -0.00507264893261
alpha =  100
mse = 0.0268483539524 mse3 = 0.0253856493086 corr =  0.516098558506 corr/f =  0.013908286217
alpha =  215
mse = 0.0260321491214 mse3 = 0.0254602448509 corr =  0.526126298582 corr/f =  0.00471485106943
alpha =  463
mse = 0.0260385725264 mse3 = 0.0254524672531 corr =  0.526189432377 corr/f =  0.00708708928762
alpha =  1000
mse = 0.0260244274835 mse3 = 0.0254725327375 corr =  0.526473479945 corr/f =  -0.0172543864282
alpha =  2150
mse = 0.0260333623156 mse3 = 0.0254737355323 corr =  0.52625890974 corr/f =  0.0031943829073
alpha =  4630
mse = 0.0260345218244 mse3 = 0.0254734792622 corr =  0.526231619277 corr/f =  -0.00609241636424
# 1000 hidden units
alpha =  10
mse = 0.0328572648596 mse3 = 0.0284966215608 corr =  0.410308254947 corr/f =  0.00435665440427
alpha =  21.5
mse = 0.0309906030262 mse3 = 0.0302436782225 corr =  0.44697193778 corr/f =  0.00510129285172
alpha =  46.3
mse = 0.0295077108661 mse3 = 0.0269230314823 corr =  0.473879171256 corr/f =  0.00251431970934
alpha =  100
mse = 0.0279110859329 mse3 = 0.026649793017 corr =  0.505861508443 corr/f =  0.00248505075961
alpha =  215
mse = 0.0260253419176 mse3 = 0.0254585741067 corr =  0.526396563133 corr/f =  0.0115736274971
alpha =  463
mse = 0.0260358905597 mse3 = 0.0254586368549 corr =  0.526244775777 corr/f =  0.00812097605867
alpha =  1000
mse = 0.0260335231486 mse3 = 0.0254591409194 corr =  0.526318120289 corr/f =  -5.22925569317e-05
alpha =  2150
mse = 0.0260322949821 mse3 = 0.0254790678555 corr =  0.526284473041 corr/f =  -0.00129931112895
alpha =  4630
mse = 0.0260327214528 mse3 = 0.0254801744005 corr =  0.526276080752 corr/f =  -0.00486408639894
# 1500 hidden units
alpha =  10
mse = 0.0333067263024 mse3 = 0.0294632805055 corr =  0.404450299217 corr/f =  0.00623424685016
alpha =  21.5
mse = 0.0314154769054 mse3 = 0.0286489954899 corr =  0.435704988828 corr/f =  0.0191877514479
alpha =  46.3
mse = 0.0299952029797 mse3 = 0.0270734514815 corr =  0.468377282514 corr/f =  0.00357957079837
alpha =  100
mse = 0.0267565477874 mse3 = 0.025783825496 corr =  0.518637644749 corr/f =  0.014583731878
alpha =  215
mse = 0.0260271656156 mse3 = 0.0255130726948 corr =  0.526410110756 corr/f =  0.00968279872684
alpha =  463
mse = 0.0260406542702 mse3 = 0.0254740956085 corr =  0.526070598651 corr/f =  0.0179831756134
alpha =  1000
mse = 0.0260299138273 mse3 = 0.0254693979994 corr =  0.526405319713 corr/f =  -0.00248935616501
alpha =  2150
mse = 0.0260315568162 mse3 = 0.0254819964988 corr =  0.526302264047 corr/f =  -0.00457320939913
alpha =  4630
mse = 0.0260325214681 mse3 = 0.0254824197501 corr =  0.526277813365 corr/f =  -0.00379869067885
# 2000 hidden units
alpha =  10
mse = 0.0343910053318 mse3 = 0.0292845948377 corr =  0.395540511406 corr/f =  -0.00329358094163
alpha =  21.5
mse = 0.0317904671887 mse3 = 0.0276352392261 corr =  0.425353744116 corr/f =  0.0042257750803
alpha =  46.3
mse = 0.0302637561095 mse3 = 0.0277676192918 corr =  0.463447935674 corr/f =  0.00444864804765
alpha =  100
mse = 0.0268106649769 mse3 = 0.0257510620581 corr =  0.518513294042 corr/f =  0.0143077127657
alpha =  215
mse = 0.0280740134585 mse3 = 0.0260639964332 corr =  0.511272063723 corr/f =  0.0077713247189
alpha =  463
mse = 0.0260390365257 mse3 = 0.0254710694216 corr =  0.526075255576 corr/f =  0.0203713076436
alpha =  1000
mse = 0.0260296475987 mse3 = 0.0254775917271 corr =  0.526398600979 corr/f =  -0.00467519392648
alpha =  2150
mse = 0.0260241839296 mse3 = 0.0254759861115 corr =  0.526497437989 corr/f =  0.00447707594765
alpha =  4630
mse = 0.026032195696 mse3 = 0.0254819046069 corr =  0.526273929195 corr/f =  0.00307244712253

# Summary for plotting
from mpl_toolkits.mplot3d import Axes3D
mse = np.array([[0.0326, 0.03041, 0.028579, 0.026486, 0.02627, 0.026025, 0.026034, 0.026034, 0.026035], [0.03311, 0.03138, 0.028772, 0.026737, 0.026257, 0.026022, 0.026034, 0.026034, 0.026034], [0.0322869, 0.031147, 0.028996, 0.0267612, 0.026239, 0.0260359, 0.0260239, 0.0260336, 0.0260340], [0.0327665, 0.0315467, 0.0294962, 0.0268483, 0.0260321, 0.0260385, 0.0260244, 0.0260333, 0.0260345], [0.0328572, 0.0309906, 0.0295077, 0.027911, 0.0260253, 0.02603589, 0.0260335, 0.0260322, 0.0260327], [0.033330, 0.031415476, 0.029995, 0.0267565, 0.026027165, 0.0260406, 0.02602991, 0.026031556, 0.0260325214], [0.034391, 0.031790, 0.03026375, 0.02681066, 0.028074, 0.02603903, 0.02602964, 0.0260241, 0.0260321]])
corr = []

xs, ys = np.meshgrid(regularization_param, hidden_layer_size)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_wireframe(np.log10(xs), ys, mse)
ax.set_xlabel('Log10(alpha)'); ax.set_ylabel('Hidden units'); ax.set_zlabel('MSE')

# Layer size is not that important as long as it is not too small or not too big (200-800)
# Best results with little layers (50,100) at alpha 463, medium layers (300, 600, 1000, 1500) at alpha 1000 and lot of layers (2000) at alpha 2150
# Stick with 400, look for the best alpha.

regularization_params = 10**np.array([2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5]) #2-4
for alpha in regularization_params:
	print('alpha = ', alpha); model = MLPRegressor(hidden_layer_sizes = (400,), alpha=alpha, max_iter=1000, random_state=123, early_stopping=True, algorithm='l-bfgs', tol = 1e-8); model.fit(X_train, y_train); y_pred = model.predict(X_test); mse = ((y_pred-y_test)**2).mean(); mse3 = ((y_pred[:,3] - y_test[:,3])**2).mean(); corr = 0; corrf = 0;
	for i in range(718):
		corr +=  np.corrcoef(y_pred[i,:], y_test[i,:])[0,1]
	for i in range(768):
		corrf +=  np.corrcoef(y_pred[:,i], y_test[:,i])[0,1]
	corr =	corr/718; corrf =	corrf/768; print('mse =', mse, 'mse3 =',mse3, 'corr =', corr, 'corr/f =', corrf)
	
# Results
alpha =  100.0
mse = 0.0273122903073 mse3 = 0.0259677794465 corr = 0.509818677762 corr/f = 0.0139174843286
alpha =  125.892541179
mse = 0.0269292134791 mse3 = 0.0258588174627 corr = 0.514948016546 corr/f = 0.0101797111697
alpha =  158.489319246
mse = 0.0261187101852 mse3 = 0.0254241900759 corr = 0.525678068293 corr/f = 0.01768006318
alpha =  199.526231497
mse = 0.0260289316828 mse3 = 0.025449590274 corr = 0.526243276356 corr/f = 0.0107668366837
alpha =  251.188643151
mse = 0.0260334052936 mse3 = 0.0254770603568 corr = 0.526163889163 corr/f = 0.00445740104598
alpha =  316.227766017
mse = 0.0260340069797 mse3 = 0.025480846556 corr = 0.526249190539 corr/f = 0.00381990488516
alpha =  398.107170553
mse = 0.02603697675 mse3 = 0.0254570959511 corr = 0.526248454401 corr/f = 0.0118820355357
alpha =  501.187233627
mse = 0.0260386648981 mse3 = 0.0254542083742 corr = 0.526208608098 corr/f = 0.00618621224792
alpha =  630.95734448
mse = 0.0260268313282 mse3 = 0.0254728248681 corr = 0.526415448042 corr/f = -0.0121991826559
alpha =  794.328234724
mse = 0.0260254667981 mse3 = 0.0254744995109 corr = 0.526454997127 corr/f = -0.0137542222462
alpha =  1000.0
mse = 0.0260246314173 mse3 = 0.0254736898827 corr = 0.526478427337 corr/f = -0.0157716618655
alpha =  1258.92541179
mse = 0.0260292401014 mse3 = 0.0254721601918 corr = 0.526366761924 corr/f = -0.0114048189136
alpha =  1584.89319246
mse = 0.0260331631979 mse3 = 0.0254753310248 corr = 0.526262869995 corr/f = 0.00145002063781
alpha =  1995.26231497
mse = 0.0260332085867 mse3 = 0.0254763965469 corr = 0.526263139262 corr/f = 0.00778444271413
alpha =  2511.88643151
mse = 0.0260335943645 mse3 = 0.0254747249329 corr = 0.526255191256 corr/f = 0.00766974906114
alpha =  3162.27766017
mse = 0.0260340124307 mse3 = 0.0254731690473 corr = 0.526246698547 corr/f = 0.00156027927876

# Results for no early stopping (may benefit from having 10% more data)
alpha =  100.0
mse = 0.0268527915082 mse3 = 0.0257736453523 corr = 0.506684294523 corr/f = 0.00489021965057
alpha =  125.892541179
mse = 0.0264837948152 mse3 = 0.0254768340204 corr = 0.514276799048 corr/f = 0.007926114599
alpha =  158.489319246
mse = 0.0261829230793 mse3 = 0.0253976839628 corr = 0.523286570917 corr/f = 0.00732928046755
alpha =  199.526231497
mse = 0.0260631689827 mse3 = 0.0253442069968 corr = 0.525469109624 corr/f = 0.0113582710533
alpha =  251.188643151
mse = 0.026014512594 mse3 = 0.0254908697433 corr = 0.526842253418 corr/f = 0.00756680160785
alpha =  316.227766017
mse = 0.026010975998 mse3 = 0.0254582927661 corr = 0.526840020671 corr/f = 0.00750018972894
alpha =  398.107170553
mse = 0.0260175530549 mse3 = 0.0254607311509 corr = 0.526687988552 corr/f = 0.00873232736361
alpha =  501.187233627
mse = 0.0260257540427 mse3 = 0.0254922325058 corr = 0.526432336307 corr/f = 0.0151675870029
alpha =  630.95734448
mse = 0.026031156003 mse3 = 0.0254807470903 corr = 0.52633222233 corr/f = 0.00199525906056
alpha =  794.328234724
mse = 0.0260263096342 mse3 = 0.0254700193532 corr = 0.526473386833 corr/f = 0.00468509989844
alpha =  1000.0
mse = 0.0260289125697 mse3 = 0.0254806165831 corr = 0.52638569475 corr/f = 0.00411384993672
alpha =  1258.92541179
mse = 0.0260292347757 mse3 = 0.0254820939947 corr = 0.52637408319 corr/f = 0.00469763622303
alpha =  1584.89319246
mse = 0.0260279402786 mse3 = 0.0254717789132 corr = 0.526413168839 corr/f = 0.00741311954988
alpha =  1995.26231497
mse = 0.0260285378426 mse3 = 0.025470744566 corr = 0.526396420141 corr/f = 0.00604793123145
alpha =  2511.88643151
mse = 0.0260352485469 mse3 = 0.0254810544836 corr = 0.526208552953 corr/f = 0.00491596135664
alpha =  3162.27766017
mse = 0.0260353613833 mse3 = 0.0254805508387 corr = 0.526212010762 corr/f = 0.00368088933633

# Results for tanh (early stopping back on)
alpha =  100.0
mse = 0.0267311994602 mse3 = 0.0255989995642 corr = 0.51260610755 corr/f = 0.00904124880602
alpha =  125.892541179
mse = 0.0265829634657 mse3 = 0.0258465737491 corr = 0.515789311162 corr/f = -0.00245773701335
alpha =  158.489319246
mse = 0.0264759850512 mse3 = 0.0257934832137 corr = 0.520741493078 corr/f = 0.00930859400444
alpha =  199.526231497
mse = 0.0263791101232 mse3 = 0.0255147324488 corr = 0.523095015555 corr/f = 0.0111139275074
alpha =  251.188643151
mse = 0.0262143992129 mse3 = 0.0254960198695 corr = 0.524453339378 corr/f = 0.0119226031889
alpha =  316.227766017
mse = 0.0262333341194 mse3 = 0.0254804427279 corr = 0.524016218251 corr/f = 0.00452360118884
alpha =  398.107170553
mse = 0.0264078581431 mse3 = 0.0257668909766 corr = 0.522368189425 corr/f = 0.00482078188357
alpha =  501.187233627
mse = 0.0263423251431 mse3 = 0.0254732757837 corr = 0.523314265818 corr/f = -0.0102770018119
alpha =  630.95734448
mse = 0.0263614765664 mse3 = 0.0256888436488 corr = 0.523157214822 corr/f = 0.00151179839274
alpha =  794.328234724
mse = 0.0263552300854 mse3 = 0.0256775772088 corr = 0.523659746469 corr/f = 0.0046600048435
alpha =  1000.0
mse = 0.0261222072071 mse3 = 0.0254613545356 corr = 0.524142516858 corr/f = -0.011343047791
alpha =  1258.92541179
mse = 0.0260953647675 mse3 = 0.0254362358614 corr = 0.524857626909 corr/f = -0.0105142323006
alpha =  1584.89319246
mse = 0.0260703505203 mse3 = 0.0255163891158 corr = 0.525398376217 corr/f = -0.0071044301137
alpha =  1995.26231497
mse = 0.0260435411581 mse3 = 0.0255342429008 corr = 0.525993654598 corr/f = 0.00213592089033
alpha =  2511.88643151
mse = 0.0260380673211 mse3 = 0.0255159059063 corr = 0.526123293069 corr/f = 0.00121380414705
alpha =  3162.27766017
mse = 0.0260353251133 mse3 = 0.0255060169512 corr = 0.526180168819 corr/f = -0.00139880504271
alpha =  3981.07170553
mse = 0.0260356615056 mse3 = 0.0255160249948 corr = 0.526237659116 corr/f = 0.0104346390182
alpha =  5011.87233627
mse = 0.0260360730329 mse3 = 0.0255258805172 corr = 0.526218401272 corr/f = 0.0141138699938
alpha =  6309.5734448
mse = 0.0260345373627 mse3 = 0.0254847678959 corr = 0.526198769685 corr/f = 0.00124047208615
alpha =  7943.28234724
mse = 0.0260352892834 mse3 = 0.0254780995715 corr = 0.526192051659 corr/f = 0.00755737018944
alpha =  10000.0
mse = 0.0260359619745 mse3 = 0.0254773851224 corr = 0.526184109318 corr/f = 0.00829815637956

# l-bfgs
alpha =  100.0
mse = 0.329827846088 mse3 = 0.306400166082 corr = 0.0981890080305 corr/f = -0.000960043497599
alpha =  125.892541179
mse = 0.0321336128987 mse3 = 0.0303072088854 corr = 0.418570435168 corr/f = 0.00100857641418
alpha =  158.489319246
mse = 0.0317281486046 mse3 = 0.0291669462651 corr = 0.426596678626 corr/f = 0.007367068969
alpha =  199.526231497
mse = 0.0316734887139 mse3 = 0.0296773399404 corr = 0.428259985061 corr/f = 0.00878383043321
alpha =  251.188643151
mse = 0.0314366943783 mse3 = 0.0295359695659 corr = 0.434231077435 corr/f = -0.000534390720376
alpha =  316.227766017













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
