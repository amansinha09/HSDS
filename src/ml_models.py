import os, time, itertools, ast
from sklearn import preprocessing
import extract_baseline_features
import utils, classifiers
import pre_processing as dproc

n_train = 300
n_test = 50

def baseline(tweets_train, train_labels, tweets_test, test_labels):
	subj_dict = dproc.get_subj_lexicon('hindi_lexicon.tff')
	types_of_features = ['1', '2','ngrams']# '3' is removed

	for t in types_of_features:

		start = time.time()
		utils.print_model_title("Classification using features type "+ t)
		if t is '1':
			x_train_features = extract_baseline_features.get_features1(tweets_train, subj_dict)
			x_test_features = extract_baseline_features.get_features1(tweets_test, subj_dict)

		if t is '2':
			x_train_features = extract_baseline_features.get_features2(tweets_train, subj_dict)
			x_test_features = extract_baseline_features.get_features2(tweets_test, subj_dict)

		#if t is '3':
		#	x_train_features = extract_baseline_features.get_feature3(tweets_train, subj_dict)
		#	x_test_features = extract_baseline_features.get_feature3(tweets_test, subj_dict)

		if t is 'ngrams':
			ngram_map, x_train_features = extract_baseline_features.get_ngram_features(tweets_train, n=1)
			x_test_features = extract_baseline_features.get_ngram_features_from_map(tweets_test, ngram_map,  n=1)

		#get the class ratio
		class_ratio = utils.get_classes_ratio_as_dict(train_labels)

		# train on a linear Support Vector Classifer
		print('\n Evaluating a linear SVM mdel...')
		classifiers.linear_svm(x_train_features, train_labels, x_test_features, test_labels, class_ratio)

		#train on logistic regression
		classifiers.logistic_regression(x_train_features, train_labels, x_test_features, test_labels, class_ratio)
		end =time.time()

		print("Completion time of the baseline model with features type %s: %.3f s = %.3f min" % (t,(end -start), (end -start)/ 60.0))


if __name__ == '__main__':

	#path = '\\'.join((os.getcwd()).split('\\')[:-1]) #for windows
	path = os.getcwd()[:os.getcwd().rfind('/')] #for unbuntu
	#print(path+'/res/datasets/riloff')

	#to_write_filename = path + '/stats/ml_analysis.txt'#change to '\\stats\\ml_analysis.txt'
	#utils.intialize_writer(to_write_filename)

	dataset = 'riloff_hn'

	'''
	train_tokens = 
	test_pos = 
	train_labels = 
	test_tokens = 
	test_pos = 
	test_labels = 
	'''
	train_tokens, train_pos, train_labels, test_tokens, test_pos, test_labels = dproc.get_dataset(dataset)
	run_baseline = True
	
	#x_train_features = extract_baseline_features.get_ngram_features(train_tokens[:5], n=1)
	#print((x_train_features))
	
	if run_baseline:
		baseline(train_tokens, train_labels, test_tokens, test_labels)
	else:
		#TODO
		print("ml_model TODO")
	


	
