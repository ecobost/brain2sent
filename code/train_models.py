# Written by: Erick Cobos T
# Date: 29-July-2016
""" Trains the many different models using either a neural network or Lasso.

The neural network has a single hidden layer with 400 units and is trained using
SGD+NAG dividing the learning rate by 5 every time the training loss converges.
For the Lasso (l1-regularized linear model), a separate model is trained for 
each element y_i in the output: a forest of 250 extremely randomized trees
fitted for regression is used to do feature selection and the model is trained 
with the selected features. See code for details.

Models use BOLD activations to predict the feature representation (image 
vectors) of the image the subject is seeing. See data_archive.md for a 
description of the input and desired output for each model.
"""
import h5py
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import GridSearchCV

def main():
	""" Main function. Trains all models. """
	
	# Lasso models (each takes around 5 days... yep)
	train_lasso('train_bold_4s_delay.h5', 'train_feats.h5', 'a')
	train_lasso('train_bold_4s_delay.h5', 'train_1sec_feats.h5', 'b')
	train_lasso('train_bold_5s_delay.h5', 'train_feats.h5', 'c')
	train_lasso('train_bold_5s_delay.h5', 'train_1sec_feats.h5', 'd')
	train_lasso('train_bold_6s_delay.h5', 'train_feats.h5', 'e')
	train_lasso('train_bold_6s_delay.h5', 'train_1sec_feats.h5', 'f')
	train_lasso('train_bold_7s_delay.h5', 'train_feats.h5', 'g')
	train_lasso('train_bold_7s_delay.h5', 'train_1sec_feats.h5', 'h')
	train_lasso('train_bold_8s_delay.h5', 'train_feats.h5', 'i')
	train_lasso('train_bold_8s_delay.h5', 'train_1sec_feats.h5', 'j')
	train_lasso('train_pd_deconv.h5', 'train_feats.h5', 'k')
	train_lasso('train_pd_deconv.h5', 'train_1sec_feats.h5', 'l')
	train_lasso('train_wu_deconv.h5', 'train_feats.h5', 'm')
	train_lasso('train_wu_deconv.h5', 'train_1sec_feats.h5', 'n')
	train_lasso('train_bold.h5', 'train_conv_feats.h5', 'o')

	# Neural network models (each takes around 8 hours)
	train_neural_network('train_bold_4s_delay.h5', 'train_feats.h5', 'p')
	train_neural_network('train_bold_4s_delay.h5', 'train_1sec_feats.h5', 'q')
	train_neural_network('train_bold_5s_delay.h5', 'train_feats.h5', 'r')
	train_neural_network('train_bold_5s_delay.h5', 'train_1sec_feats.h5', 's')
	train_neural_network('train_bold_6s_delay.h5', 'train_feats.h5', 't')
	train_neural_network('train_bold_6s_delay.h5', 'train_1sec_feats.h5', 'u')
	train_neural_network('train_bold_7s_delay.h5', 'train_feats.h5', 'v')
	train_neural_network('train_bold_7s_delay.h5', 'train_1sec_feats.h5', 'w')
	train_neural_network('train_bold_8s_delay.h5', 'train_feats.h5', 'x')
	train_neural_network('train_bold_8s_delay.h5', 'train_1sec_feats.h5', 'y')
	train_neural_network('train_pd_deconv.h5', 'train_feats.h5', 'z')
	train_neural_network('train_pd_deconv.h5', 'train_1sec_feats.h5', 'aa')
	train_neural_network('train_wu_deconv.h5', 'train_feats.h5', 'ab')
	train_neural_network('train_wu_deconv.h5', 'train_1sec_feats.h5', 'ac')
	train_neural_network('train_bold.h5', 'train_conv_feats.h5', 'ad')


def train_neural_network(features_filename, targets_filename, model_name):
	""" Trains a neural network with 400 units in the hidden layer. """
	print('Training model', model_name)
	
	# Read regression features
	Xs_file = h5py.File(features_filename, 'r')
	Xs = np.array(Xs_file['responses'])
	Xs_file.close()

	# Read regression targets
	y_file = h5py.File(targets_filename, 'r')
	y = np.array(y_file['feats'])
	y_file.close()

	# Drop first and last 10 samples to avoid convolution artifacts
	Xs = Xs[10:-10, :]
	y = y[10:-10, :]

	# Train model (searching for best regularization parameter)
	nn = MLPRegressor(hidden_layer_sizes=(400,), algorithm='sgd', 
					  learning_rate='adaptive', random_state=123, tol=1e-8)
	regularization_params = {'alpha': np.logspace(2, 3, 20)}
	cv = GridSearchCV(nn, regularization_params, cv=5, n_jobs=-1).fit(Xs, y)
	model = cv.best_estimator_
		
	# Print cross-validation results
	print('CV results: ', cv.results_['test_mean_score'])
	print('Best alpha:', cv.best_params_['alpha'], 'R2 score:', cv.best_score_)

	# Save model
	model_file = h5py.File('model_' + model_name + '.h5', 'w')
	model_file.create_dataset('weights_ih', data=model.coefs_[0].transpose())
	model_file.create_dataset('bias_ih', data=model.intercepts_[0])
	model_file.create_dataset('weights_ho', data=model.coefs_[1].transpose())
	model_file.create_dataset('bias_ho', data=model.intercepts_[1])
	model_file.close()

	
def train_lasso(features_filename, targets_filename, model_name):
	""" Does feature selection using a random forest and fits a Lasso model. """
	print('Training model', model_name)
	
	# Read regression features
	Xs_file = h5py.File(features_filename, 'r')
	Xs = np.array(Xs_file['responses'])
	Xs_file.close()

	# Read regression targets
	y_file = h5py.File(targets_filename, 'r')
	y = np.array(y_file['feats'])
	y_file.close()

	# Drop first and last 10 samples to avoid convolution artifacts
	Xs = Xs[10:-10, :]
	y = y[10:-10, :]
	
	# Initialize containers for weights, bias and bookkeeping
	num_features = Xs.shape[1]
	num_outputs = y.shape[1]
	weights = np.zeros((num_outputs, num_features))
	bias = np.zeros(num_outputs)
	best_models = {'mse': np.full(num_outputs, np.inf),
				   'alpha': np.zeros(num_outputs), 'k': np.zeros(num_outputs)}
	
	# Over every output element
	for i in range(num_outputs):
		y_i = y[:, i]
		
		# Train feature selector (random forest)
		selector = ExtraTreesRegressor(n_estimators=250, max_features=1000,
									   random_state=123, n_jobs=-1).fit(Xs, y_i)
		sorted_indices = selector.feature_importances_.argsort()
									   
		# Train models with different number of features
		for k in [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 200, 300, 500, 1000]:
			# Select best k features
			selected_features = sorted_indices[-k:]
			Xs_select = Xs[:, selected_features]

			# Train model
			model = LassoCV(selection='random', random_state=123, cv=5,
							n_jobs=-1).fit(Xs_select, y_i)
			model_mse = model.mse_path_.mean(axis=1).max()
			
			# If best model, save it
			if model_mse < best_models['mse'][i]:
				best_models['mse'][i] = model_mse
				best_models['alpha'][i] = model.alpha_
				best_models['k'][i] = k
				weights[i, selected_features] = model.coef_
				bias[i] = model.intercept_
				
		# Report and save checkpoint
		if i%100 == 0:
			print(i+1, 'out of', num_outputs)
			print('MSE:', best_models['mse'][:i])
			print('Alphas:', best_models['alpha'][:i])
			print('Ks:', best_models['k'][:i])
						
			print('Saving checkpoint...')
			checkpoint_name = 'model_' + model_name + '_' + str(i) + '.h5'
			model_file = h5py.File(checkpoint_name, 'w')
			model_file.create_dataset('weights', data=weights)
			model_file.create_dataset('bias', data=bias)
			model_file.close()

	# Print final cross-validation results
	print('Final results')
	print(i+1, 'out of', num_outputs)
	print('MSE:', best_models['mse'])
	print('Alphas:', best_models['alpha'])
	print('Ks:', best_models['k'][:i])
	print('Sparsity (per output):', (weights != 0).sum(axis=1))
	print('Sparsity (per feature):', (weights != 0).sum(axis=0))
		
	# Save model
	model_file = h5py.File('model_' + model_name + '.h5', 'w')
	model_file.create_dataset('weights', data=weights)
	model_file.create_dataset('bias', data=bias)
	model_file.close()
	
if __name__ == "__main__":
	main()
