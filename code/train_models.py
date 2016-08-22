# Written by: Erick Cobos T
# Date: 05-August-2016
""" Trains the many different models used to predict image vectors.

Models use BOLD activations to predict the feature representation (image 
vectors) of the image the subject is seeing. See data_archive.md for a 
description of each model and its inputs. 

Important note:
I very much recommend you install OpenBlas for training. It gives me close to a 
70x speed-up. Some models would take weeks otherwise. You can do:
	sudo apt-get remove libatlas3gf-base libatlas-dev
	sudo apt-get install libopenblas-dev liblapack-dev
	sudo pip3 install --upgrade numpy
"""
import h5py
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_selection import f_regression
from sklearn.neural_network import MLPRegressor

def main():
	""" Main function. Trains all models. """
	# Ridge models (each takes around 4 days)
	train_ridge('train_4sec_bold.h5', 'train_feats.h5', 'l2_4sec')
	train_ridge('train_5sec_bold.h5', 'train_feats.h5', 'l2_5sec')
	train_ridge('train_6sec_bold.h5', 'train_feats.h5', 'l2_6sec')
	train_ridge('train_7sec_bold.h5', 'train_feats.h5', 'l2_7sec')
	train_ridge('train_4sec_smooth_bold.h5', 'train_feats.h5', 'l2_4sec_smooth')
	train_ridge('train_5sec_smooth_bold.h5', 'train_feats.h5', 'l2_5sec_smooth')
	train_ridge('train_6sec_smooth_bold.h5', 'train_feats.h5', 'l2_6sec_smooth')
	train_ridge('train_7sec_smooth_bold.h5', 'train_feats.h5', 'l2_7sec_smooth')
	train_ridge('train_bold.h5', 'train_conv_feats.h5', 'l2_conv')
	train_ridge('train_smooth_bold.h5', 'train_conv_feats.h5', 'l2_conv_smooth')
	train_ridge('train_deconv.h5', 'train_feats.h5', 'l2_deconv')
		
	# F-statisc models (each takes around a week)
	train_f_selection('train_4sec_bold.h5', 'train_feats.h5', 'f_4sec')
	train_f_selection('train_5sec_bold.h5', 'train_feats.h5', 'f_5sec')
	train_f_selection('train_6sec_bold.h5', 'train_feats.h5', 'f_6sec')
	train_f_selection('train_7sec_bold.h5', 'train_feats.h5', 'f_7sec')
	train_f_selection('train_4sec_smooth_bold.h5', 'train_feats.h5', 'f_4sec_smooth')
	train_f_selection('train_5sec_smooth_bold.h5', 'train_feats.h5', 'f_5sec_smooth')
	train_f_selection('train_6sec_smooth_bold.h5', 'train_feats.h5', 'f_6sec_smooth')
	train_f_selection('train_7sec_smooth_bold.h5', 'train_feats.h5', 'f_7sec_smooth')
	train_f_selection('train_bold.h5', 'train_conv_feats.h5', 'f_conv')
	train_f_selection('train_smooth_bold.h5', 'train_conv_feats.h5', 'f_conv_smooth')
	train_f_selection('train_deconv.h5', 'train_feats.h5', 'f_deconv')
		
	# ROI models (each takes around 2 days)
	train_roi_selection('train_4sec_bold.h5', 'train_feats.h5', 'roi_4sec')
	train_roi_selection('train_5sec_bold.h5', 'train_feats.h5', 'roi_5sec')
	train_roi_selection('train_6sec_bold.h5', 'train_feats.h5', 'roi_6sec')
	train_roi_selection('train_7sec_bold.h5', 'train_feats.h5', 'roi_7sec')
	train_roi_selection('train_4sec_smooth_bold.h5', 'train_feats.h5', 'roi_4sec_smooth')
	train_roi_selection('train_5sec_smooth_bold.h5', 'train_feats.h5', 'roi_5sec_smooth')
	train_roi_selection('train_6sec_smooth_bold.h5', 'train_feats.h5', 'roi_6sec_smooth')
	train_roi_selection('train_7sec_smooth_bold.h5', 'train_feats.h5', 'roi_7sec_smooth')
	train_roi_selection('train_bold.h5', 'train_conv_feats.h5', 'roi_conv')
	train_roi_selection('train_smooth_bold.h5', 'train_conv_feats.h5', 'roi_conv_smooth')
	train_roi_selection('train_deconv.h5', 'train_feats.h5', 'roi_deconv')
	
	# Neural network models (each takes around 11 hours)
	train_neural_network('train_4sec_bold.h5', 'train_feats.h5', 'nn_4sec')
	train_neural_network('train_5sec_bold.h5', 'train_feats.h5', 'nn_5sec')
	train_neural_network('train_6sec_bold.h5', 'train_feats.h5', 'nn_6sec')
	train_neural_network('train_7sec_bold.h5', 'train_feats.h5', 'nn_7sec')
	train_neural_network('train_4sec_smooth_bold.h5', 'train_feats.h5', 'nn_4sec_smooth')
	train_neural_network('train_5sec_smooth_bold.h5', 'train_feats.h5', 'nn_5sec_smooth')
	train_neural_network('train_6sec_smooth_bold.h5', 'train_feats.h5', 'nn_6sec_smooth')
	train_neural_network('train_7sec_smooth_bold.h5', 'train_feats.h5', 'nn_7sec_smooth')
	train_neural_network('train_bold.h5', 'train_conv_feats.h5', 'nn_conv')
	train_neural_network('train_smooth_bold.h5', 'train_conv_feats.h5', 'nn_conv_smooth')
	train_neural_network('train_deconv.h5', 'train_feats.h5', 'nn_deconv')
	
def read_inputs(features_filename, targets_filename):
	# Read regression features
	with h5py.File(features_filename, 'r') as Xs_file:
		Xs = np.array(Xs_file['responses'])

	# Read regression targets
	with h5py.File(targets_filename, 'r') as y_file:
		y = np.array(y_file['feats'])
	
	# Drop first and last 10 samples to avoid convolution/smoothing artifacts
	Xs = Xs[10:-10, :]
	y = y[10:-10, :]
	
	return Xs, y
	
def save_linear_model(model_filename, weights, bias):
	with h5py.File(model_filename, 'w') as model_file:
		model_file.create_dataset('weights', data=weights)
		model_file.create_dataset('bias', data=bias)
	
	
def train_lasso(features_filename, targets_filename, model_name):
	""" Trains a Lasso model on all input features (no feature selection). """
	print('Training model', model_name)
	
	# Read features and targets
	Xs, y = read_inputs(features_filename, targets_filename)
	
	# Initialize containers for weights and bias
	num_features = Xs.shape[1]
	num_outputs = y.shape[1]
	weights = np.zeros((num_outputs, num_features))
	bias = np.zeros(num_outputs)
	
	# And for bookkeeping
	best_models = {'R2': np.zeros(num_outputs), 'alpha': np.zeros(num_outputs),
				   'sparsity': np.zeros(num_outputs)}
	
	# Over every output element
	for i in range(num_outputs):
		y_i = y[:, i]
		
		# Train model (searching for best regularization parameter)
		cv = GridSearchCV(Lasso(), regularization_params, cv=5, n_jobs=-1).fit(Xs, y_i)
		model = cv.best_estimator_
		
		# Store it
		weights[i, :] = model.coef_
		bias[i] = model.intercept_
		
		# Bookkeping
		best_models['R2'][i] = cv.best_score_
		best_models['alpha'][i] = model.alpha
		best_models['sparsity'][i] = (model.coef_ != 0).sum()
		
		# Report and save checkpoint
		if i%100 == 0:
			print(i+1, 'out of', num_outputs)
			print('R2:', best_models['R2'][:i])
			print('Alphas:', best_models['alpha'][:i])
			print('Sparsity:', best_models['sparsity'][:i])
						
			print('Saving checkpoint...')
			checkpoint_name = model_name + '_' + str(i) + '.h5'
			save_linear_model(checkpoint_name, weights, bias)

	# Print final cross-validation results
	print('Final results')
	print(i+1, 'out of', num_outputs)
	print('R2:', best_models['R2'])
	print('Average R2:', best_models['R2'].mean()) 
	print('Alphas:', best_models['alpha'])
	print('Sparsity (per output):', (weights != 0).sum(axis=1))
	print('Sparsity (per feature):', (weights != 0).sum(axis=0))
		
	# Save model
	save_linear_model(model_name + '.h5', weights, bias)

	
def train_ridge(features_filename, targets_filename, model_name):
	""" Trains a Ridge model on all input features (no feature selection). """
	print('Training model', model_name)
		
	# Read features and targets
	Xs, y = read_inputs(features_filename, targets_filename)
	
	# Set regularization parameters
	regularization_params = {'alpha': np.logspace(3.5, 5.5, 20)}
	
	# Initialize containers for weights and bias
	num_features = Xs.shape[1]
	num_outputs = y.shape[1]
	weights = np.zeros((num_outputs, num_features))
	bias = np.zeros(num_outputs)
	
	# And for bookkeeping
	best_models = {'R2': np.zeros(num_outputs), 'alpha': np.zeros(num_outputs)}
	
	# Over every output element
	for i in range(num_outputs):
		y_i = y[:, i]
		
		# Train model (searching for best regularization parameter)
		cv = GridSearchCV(Ridge(), regularization_params, cv=5, n_jobs=-1).fit(Xs, y_i)
		model = cv.best_estimator_
		
		# Store it
		weights[i, :] = model.coef_
		bias[i] = model.intercept_
		
		# Bookkeping
		best_models['R2'][i] = cv.best_score_
		best_models['alpha'][i] = model.alpha
				
		# Report and save checkpoint
		if i%100 == 0:
			print(i+1, 'out of', num_outputs)
			print('R2:', best_models['R2'][:i])
			print('Alphas:', best_models['alpha'][:i])
						
			print('Saving checkpoint...')
			checkpoint_name = model_name + '_' + str(i) + '.h5'
			save_linear_model(checkpoint_name, weights, bias)

	# Print final cross-validation results
	print('Final results')
	print(i+1, 'out of', num_outputs)
	print('R2:', best_models['R2'])
	print('Average R2:', best_models['R2'].mean()) 
	print('Alphas:', best_models['alpha'])
		
	# Save model
	save_linear_model(model_name + '.h5', weights, bias)
	
	
def train_f_selection(features_filename, targets_filename, model_name):
	""" Does feature selection using an F-test and fits a Ridge model. """
	print('Training model', model_name)
	
	# Read features and targets
	Xs, y = read_inputs(features_filename, targets_filename)
	
	# Set regularization parameters and number of features
	regularization_params = {'alpha': np.logspace(2.5, 5, 25)}
	ks = [5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000]
	
	# Initialize containers for weights and bias
	num_features = Xs.shape[1]
	num_outputs = y.shape[1]
	weights = np.zeros((num_outputs, num_features))
	bias = np.zeros(num_outputs)
	
	# And for bookkeeping
	best_models = {'R2': np.full(num_outputs, -np.inf), 
				   'alpha': np.zeros(num_outputs), 'k': np.zeros(num_outputs)}
	
	# Over every output element
	for i in range(num_outputs):
		y_i = y[:, i]
		
		# Train feature selector
		_, p_values = f_regression(Xs, y_i)
		sorted_indices = p_values.argsort()
		
		# Number of features with p < 0.01 and p < 0.001
		p_value_ks = [(p_values < 0.01).sum(), (p_values < 0.001).sum()]
		p_value_ks = [x for x in p_value_ks if x!=0]
		   
		# Train models with different number of features
		for k in sorted(ks + p_value_ks):
			# Select best k features
			selected_features = sorted_indices[:k]  
			Xs_select = Xs[:, selected_features]
			
			# Train model (searching for best regularization parameter)
			cv = GridSearchCV(Ridge(), regularization_params, cv=5, n_jobs=-1).fit(Xs_select, y_i)
			model = cv.best_estimator_
			
			# If best model yet, store it
			if cv.best_score_ > best_models['R2'][i]:
				weights[i, :] = 0
				weights[i, selected_features] = model.coef_
				bias[i] = model.intercept_
				best_models['R2'][i] = cv.best_score_
				best_models['alpha'][i] = model.alpha
				best_models['k'][i] = k		
				
		# Report and save checkpoint
		if i%100 == 0:
			print(i+1, 'out of', num_outputs)
			print('R2:', best_models['R2'][:i])
			print('Alphas:', best_models['alpha'][:i])
			print('Ks:', best_models['k'][:i])
						
			print('Saving checkpoint...')
			checkpoint_name = model_name + '_' + str(i) + '.h5'
			save_linear_model(checkpoint_name, weights, bias)

	# Print final cross-validation results
	print('Final results')
	print(i+1, 'out of', num_outputs)
	print('R2:', best_models['R2'])
	print('Average R2:', best_models['R2'].mean()) 
	print('Alphas:', best_models['alpha'])
	print('Ks:', best_models['k'])
	print('Average K:', best_models['k'].mean())
		
	# Save model
	save_linear_model(model_name + '.h5', weights, bias)
	
	
def train_roi_selection(features_filename, targets_filename, model_name):
	""" Does feature selection using an F-test and fits a Ridge model. """
	print('Training model', model_name)
	
	# Read features and targets
	Xs, y = read_inputs(features_filename, targets_filename)
	
	# Read ROI assignments
	with h5py.File('roi_info.h5', 'r') as roi_file:
		roi = np.array(roi_file['rois'])

	# Get masks
	masks = {}
	masks['v1'] = (roi == 19) | (roi == 20)
	masks['v2'] = (roi == 21) | (roi == 22)
	masks['v3'] = (roi == 27) | (roi == 28)
	masks['v3a'] = (roi == 23) | (roi == 24)
	masks['v3b'] = (roi == 25) | (roi == 26)
	masks['v4'] = (roi == 29) | (roi == 30)
	masks['lo'] = (roi == 17) | (roi == 18)
	masks['mt'] = (roi == 5) | (roi == 6) | (roi == 7) | (roi == 8) #v5
	masks['ip'] = (roi == 3) | (roi == 4)
	masks['vo'] = (roi == 15) | (roi == 16)
	masks['OBJ'] = (roi == 9) | (roi == 10) #?
	masks['rsc'] = (roi == 13)
	masks['sts'] = (roi == 14)
	
	# Set regularization parameters
	regularization_params = {'alpha': np.logspace(2.5, 5, 25)}
	
	# Initialize containers for weights and bias
	num_features = Xs.shape[1]
	num_outputs = y.shape[1]
	weights = np.zeros((num_outputs, num_features))
	bias = np.zeros(num_outputs)
	
	# And for bookkeeping
	best_models = {'R2': np.full(num_outputs, -np.inf), 
				   'alpha': np.zeros(num_outputs), 'roi': [None]*num_outputs}
	
	# Over every output element
	for i in range(num_outputs):
		y_i = y[:, i]
		
		# Train models for different ROIs
		for ROI_name, mask in masks.items():
			# Select ROI voxels
			Xs_select = Xs[:, mask]
			
			# Train model (searching for best regularization parameter)
			cv = GridSearchCV(Ridge(), regularization_params, cv=5, n_jobs=-1).fit(Xs_select, y_i)
			model = cv.best_estimator_
			
			# If best model yet, store it
			if cv.best_score_ > best_models['R2'][i]:
				weights[i, :] = 0
				weights[i, mask] = model.coef_
				bias[i] = model.intercept_
				best_models['R2'][i] = cv.best_score_
				best_models['alpha'][i] = model.alpha
				best_models['roi'][i] = ROI_name		
				
		# Report and save checkpoint
		if i%100 == 0:
			print(i+1, 'out of', num_outputs)
			print('R2:', best_models['R2'][:i])
			print('Alphas:', best_models['alpha'][:i])
			print('ROIs:', best_models['roi'][:i])
						
			print('Saving checkpoint...')
			checkpoint_name = model_name + '_' + str(i) + '.h5'
			save_linear_model(checkpoint_name, weights, bias)

	# Print final cross-validation results
	print('Final results')
	print(i+1, 'out of', num_outputs)
	print('R2:', best_models['R2'])
	print('Average R2:', best_models['R2'].mean()) 
	print('Alphas:', best_models['alpha'])
	print('ROIs:', best_models['roi'])
		
	# Save model
	save_linear_model(model_name + '.h5', weights, bias)
	
	
def train_lasso_selection(features_filename, targets_filename, selector_filename,
						  model_name):
	""" Feature selection using the pretrained Lasso and fits a Ridge model. """
	print('Training model', model_name)
	
	# Read features and targets
	Xs, y = read_inputs(features_filename, targets_filename)
	
	# Read pretrained model
	with h5py.File(selector_filename, 'r') as selector_file:
		selector = np.array(selector_file['weights'])
	
	# Set regularization parameters and number of features
	regularization_params = {'alpha': np.logspace(1, 4, 30)}
	
	# Initialize containers for weights and bias
	num_features = Xs.shape[1]
	num_outputs = y.shape[1]
	weights = np.zeros((num_outputs, num_features))
	bias = np.zeros(num_outputs)
	
	# And for bookkeeping
	best_models = {'R2': np.zeros(num_outputs), 'alpha': np.zeros(num_outputs)}
	
	# Over every output element
	for i in range(num_outputs):
		y_i = y[:, i]
		
		# Extract feature selector
		feature_importances = abs(selector[i, :])
		mask = (feature_importances > 0)
				
		# Select features
		Xs_select = Xs[:, mask]
		
		# Train model (searching for best regularization parameter)
		cv = GridSearchCV(Ridge(), regularization_params, cv=5, n_jobs=-1).fit(Xs_select, y_i)
		model = cv.best_estimator_
		
		# Store it	
		weights[i, mask] = model.coef_
		bias[i] = model.intercept_
		
		# Bookkeeping
		best_models['R2'][i] = cv.best_score_
		best_models['alpha'][i] = model.alpha
		
		# Report and save checkpoint
		if i%100 == 0:
			print(i+1, 'out of', num_outputs)
			print('R2:', best_models['R2'][:i])
			print('Alphas:', best_models['alpha'][:i])
						
			print('Saving checkpoint...')
			checkpoint_name = model_name + '_' + str(i) + '.h5'
			save_linear_model(checkpoint_name, weights, bias)

	# Print final cross-validation results
	print('Final results')
	print(i+1, 'out of', num_outputs)
	print('R2:', best_models['R2'])
	print('Average R2:', best_models['R2'].mean()) 
	print('Alphas:', best_models['alpha'])
	print('Sparsity (per output):', (weights != 0).sum(axis=1))
	print('Sparsity (per feature):', (weights != 0).sum(axis=0))
		
	# Save model
	save_linear_model(model_name + '.h5', weights, bias)
	
	
def train_ridge_selection(features_filename, targets_filename, selector_filename,
						  model_name):
	""" Feature selection using the pretrained Ridge and fits a Ridge model. """
	print('Training model', model_name)
	
	# Read features and targets
	Xs, y = read_inputs(features_filename, targets_filename)
	
	# Read pretrained model
	with h5py.File(selector_filename, 'r') as selector_file:
		selector = np.array(selector_file['weights'])
	
	# Set regularization parameters and number of features
	regularization_params = {'alpha': np.logspace(2.5, 5, 25)}
	ks = [5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000]
	
	# Initialize containers for weights and bias
	num_features = Xs.shape[1]
	num_outputs = y.shape[1]
	weights = np.zeros((num_outputs, num_features))
	bias = np.zeros(num_outputs)
	
	# And for bookkeeping
	best_models = {'R2': np.full(num_outputs, -np.inf), 
				   'alpha': np.zeros(num_outputs), 'k': np.zeros(num_outputs)}
	
	# Over every output element
	for i in range(num_outputs):
		y_i = y[:, i]
		
		# Extract feature selector
		feature_importances = abs(selector[i, :])
		sorted_indices = feature_importances.argsort()
				   
		# Train models with different number of features
		for k in ks:
			# Select best k features
			selected_features = sorted_indices[-k:]  
			Xs_select = Xs[:, selected_features]
			
			# Train model (searching for best regularization parameter)
			cv = GridSearchCV(Ridge(), regularization_params, cv=5, n_jobs=-1).fit(Xs_select, y_i)
			model = cv.best_estimator_
			
			# If best model yet, store it
			if cv.best_score_ > best_models['R2'][i]:
				weights[i, :] = 0
				weights[i, selected_features] = model.coef_
				bias[i] = model.intercept_
				best_models['R2'][i] = cv.best_score_
				best_models['alpha'][i] = model.alpha
				best_models['k'][i] = k		
				
		# Report and save checkpoint
		if i%100 == 0:
			print(i+1, 'out of', num_outputs)
			print('R2:', best_models['R2'][:i])
			print('Alphas:', best_models['alpha'][:i])
			print('Ks:', best_models['k'][:i])
						
			print('Saving checkpoint...')
			checkpoint_name = model_name + '_' + str(i) + '.h5'
			save_linear_model(checkpoint_name, weights, bias)

	# Print final cross-validation results
	print('Final results')
	print(i+1, 'out of', num_outputs)
	print('R2:', best_models['R2'])
	print('Average R2:', best_models['R2'].mean()) 
	print('Alphas:', best_models['alpha'])
	print('Ks:', best_models['k'])
	print('Average K:', best_models['k'].mean()) 
		
	# Save model
	save_linear_model(model_name + '.h5', weights, bias)

	
def train_neural_network(features_filename, targets_filename, model_name):
	""" Trains a neural network with 400 units in the hidden layer. """
	print('Training model', model_name)
	
	# Read features and targets
	Xs, y = read_inputs(features_filename, targets_filename)
	
	# Set regularization parameters and number of features
	regularization_params = {'alpha': np.logspace(1.5, 2.5, 20)}

	# Train model (searching for best regularization parameter)
	nn = MLPRegressor(hidden_layer_sizes=(400,), algorithm='sgd', 
					  learning_rate='adaptive', random_state=123, tol=1e-8)
	cv = GridSearchCV(nn, regularization_params, cv=5, n_jobs=-1).fit(Xs, y)
	model = cv.best_estimator_
		
	# Print cross-validation results
	print('CV results: ', cv.results_['test_mean_score'])
	print('Best alpha:', cv.best_params_['alpha'], 'R2 score:', cv.best_score_)

	# Save model
	with h5py.File(model_name + '.h5', 'w') as model_file:
		model_file.create_dataset('weights_ih', data=model.coefs_[0].transpose())
		model_file.create_dataset('bias_ih', data=model.intercepts_[0])
		model_file.create_dataset('weights_ho', data=model.coefs_[1].transpose())
		model_file.create_dataset('bias_ho', data=model.intercepts_[1])
	
if __name__ == "__main__":
	main()
