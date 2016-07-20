# Written by: Erick Cobos T
# Date: 20-July-2016
""" Fits the many different models using l1-regularized linear regression. """
from scikit-learn import Lasso

# for each model:
# read responses and feats
# Choose the lambda regularization parameter
# DO k-fold crossvalidation to asses accuracy
# print accuracy
# fit the model with entire data
# save model weights
