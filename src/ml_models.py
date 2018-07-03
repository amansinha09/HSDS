import os, time, itertools
from sklearn import preprocessing
import extract_baseline_features
#import utils, classifers
#import data_processing as dproc

n_train = 300
n_test = 50

def baseline(tweets_train, train_labels, tweets_test, test_labels):
	subj_dict = dproc.get_subj_lexicon()
	types_of_features = ['1', '2', '3', 'ngrams']

	fro t in types_of_features:
	start = time.time()
	utils.print_model_title("Classification using features type "+ t)
	if t is '1':
		x_train_features = extract_baseline_features.get_feature1(tweets_train, subj_dict)
		x_test_features = extract_baseline_features.get_feature1(tweets_test, subj_dict)

	if t is '2':
		x_train_features = extract_baseline_features.get_feature2(tweets_train, subj_dict)
		x_test_features = extract_baseline_features.get_feature2(tweets_test, subj_dict)

	if t is '3':
		x_train_features = extract_baseline_features.get_feature3(tweets_train, subj_dict)
		x_test_features = extract_baseline_features.get_feature3(tweets_test, subj_dict)

	if t is 'ngrams':
		x_train_features = extract_baseline_features.get_ngram_features(tweets_train, subj_dict)
		x_test_features = extract_baseline_features.get_ngram_features(tweets_test, subj_dict)

	#get the class ratio
	class_ratio = utils.get_classes_ratio_as_dict(train_labels)

	# train on a linear Support Vector Classifer
	print('\n Evaluating a linear SVM mdel...')
	classifers.linear_svm(x_train_features, train_labels, x_test_features, test_labels, class_ratio)

	#train on logistic regression
	classifers.logistic_regression(x_train_features, train_labels, x_test_features, test_labels, class_ratio)
	end =time.time()

	print("Completion time of the baseline model with features type %s: %.3f s = %.3f min" % (t, (end -start)/ 60.0))


if __name__ == '__main__':

	path = '\\'.join((os.getcwd()).split('\\')[:-1])
	to_write_filename = path + '\\output.txtx'#change to '\\stats\\ml_analysis.txt'
	utils.intialize_writer(to_write_filename)

	dataset = 'riloff'
	train_tokens, train_pos, train_labels, test_tokens, test_pos, test_labels = dproc.get_data(dataset)
	run_baseline = False
	if run_baseline:
		baseline(train_tokens, train_labels, test_tokens, test_labels)
	else:
		#TODO
		print("ml_model TODO")



