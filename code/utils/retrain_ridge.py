# Written by: Erick Cobos T. (a01184587@itesm.mx)
# Date: 12-Aug-2016
""" Script to retrain the l2 (Ridge) models with bigger alphas for those 
features who selected the highest alpha in the original range.

For an alpha to be that high usually means that the model isn't learning
anything and we would do better just predicting the mean. Results do not improve
much as 316227 is already close to the biggest desired alpha

I put all r2 scores in a space separated file called 'r2.csv' and all alphas in
a space separated file called 'alphas.csv'. 
"""
import csv
import numpy as np
import h5py
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Ridge

# Set params
highest_alpha = 316227
model_filename = 'l2_4sec.h5'
features_filename = 'train_4sec_bold.h5'
targets_filename = 'train_feats.h5'
regularization_params = {'alpha': np.logspace(5.5, 6, 6)}

# Read all R^2 scores
r2_values = [] # empty list
with open('r2.csv', newline='') as csv_file:
	reader = csv.reader(csv_file, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
	for row in reader:
		r2_values.extend(row)
r2_values = list(filter(lambda x: x!= '', r2_values))
r2_values = np.array(r2_values) 

# Read all alphas (same as above)
alphas = [] # empty list
with open('alphas.csv', newline='') as csv_file:
	reader = csv.reader(csv_file, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
	for row in reader:
		alphas.extend(row)
alphas = list(filter(lambda x: x!= '', alphas))
alphas = np.array(alphas)

# Read model
with h5py.File(model_filename, 'r') as model_file:
	weights = np.array(model_file['weights'])
	bias = np.array(model_file['bias'])

# Read regression features
with h5py.File(features_filename, 'r') as Xs_file:
	Xs = np.array(Xs_file['responses'])

# Read regression targets
with h5py.File(targets_filename, 'r') as y_file:
	y = np.array(y_file['feats'])

# Drop first and last 10 samples to avoid convolution/smoothing artifacts
Xs = Xs[10:-10, :]
y = y[10:-10, :]

# What output units need to be retrained
to_retrain = (alphas >= highest_alpha)
print('Initial R2', r2_values.mean())
print(str(to_retrain.sum()), 'units will be retrained')

# Retrain
for i in to_retrain.nonzero():
	y_i = y[:, i]
		
	# Train model (searching for best regularization parameter)
	cv = GridSearchCV(Ridge(), regularization_params, cv=5, n_jobs=-1).fit(Xs, y_i)
	model = cv.best_estimator_
	
	# Store it
	weights[i, :] = model.coef_
	bias[i] = model.intercept_
	
	# Bookkeping
	r2_values[i] = cv.best_score_
	alphas[i] = model.alpha

# Print final cross-validation results
print('Final results')
print('R2:', r2_values)
print('Average R2:', r2_values.mean())
print('Variance weighted R2:', np.average(r2_values, weights=y.var(axis=0))) #*
print('Alphas:', alphas)
	
# Save model
with h5py.File('retrained_' + model_filename, 'w') as model_file:
	model_file.create_dataset('weights', data=weights)
	model_file.create_dataset('bias', data=bias)

# *Not quite right cause the weighted r2 should be calculated with the variance
# from each of the 5 held out cross-validation sets and then averaged.
# Probably a good approximation. Only used to compare with nn models.
