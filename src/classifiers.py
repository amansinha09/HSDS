#classifiers 
#[1802 hrs 030718]
import utils
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE


#linear svm
#logistic regression

def linear_svm(x_train, y_train, x_test, y_test, class_ratio= 'balanced'):
	utils.print_model_title("Linear SVM")
	svm  = LinearSVC(C =0.01, class_weight =class_ratio, penalty = 'l2')
	svm.fit(x_train, y_train)
	y_hat = svm.predict(x_test)
	utils.print_statistics(y_test, y_hat)

def logistic_regression(x_train, y_train, x_test, y_test, class_ratio='balanced'):
	utils.print_model_title("Logistic Regression")
	regr = LogisticRegression(C= 0.01, class_weight = class_ratio, penalty='l2')
	regr.fit(x_train, y_train)
	y_hat = regr.predict(x_test)
	utils.print_statistics(y_test, y_hat)
