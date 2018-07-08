import os, time, itertools, ast
from sklearn import preprocessing
import extract_baseline_features
import extract_ml_features as emf 
import utils, classifiers
import pre_processing as dproc
import test as t
n_train = 300
n_test = 50

pragmatic= True
lexical = True
pos_grams = True
sentiment =False
topic = False
similarity =True
pos_ngram_list =[1]
ngram_list =[1]
embedding_dim =100
word2vec_map = t.use_w2v()



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
		print('\n Evaluating a linear SVM model...')
		classifiers.linear_svm(x_train_features, train_labels, x_test_features, test_labels, class_ratio)

		#train on logistic regression
		classifiers.logistic_regression(x_train_features, train_labels, x_test_features, test_labels, class_ratio)
		end =time.time()

		print("Completion time of the baseline model with features type %s: %.3f s = %.3f min" % (t,(end -start), (end -start)/ 60.0))



def ml_model(train_tokens, train_pos, y_train, test_tokens, test_pos, y_test):

	print("Processing TRAIN SET features...\n")
	start = time.time()
	train_pragmatic, train_lexical, train_pos, train_sim = emf.get_feature_set(train_tokens, train_pos, pragmatic= pragmatic, lexical =lexical, ngram_list = ngram_list, pos_grams = pos_grams, pos_ngram_list= pos_ngram_list,
		sentiment = sentiment, topic =topic, similarity= similarity, word2vec_map = word2vec_map)
	end = time.time()
	
	print("Completion time of extracting train models: %.3f s = %.3f min" % ((end -start), (end - start)/60.0))

	print("Processing TEST SET features ...\n")
	start =time.time()
	test_pragmatic, test_lexical, test_pos, test_sim = emf.get_feature_set(test_tokens, test_pos, pragmatic= pragmatic, lexical =lexical, ngram_list = ngram_list, pos_grams = pos_grams, pos_ngram_list= pos_ngram_list,
		sentiment = sentiment, topic =topic, similarity= similarity, word2vec_map = word2vec_map)
	end = time.time()
	print("Completion time of extracting test models: %.3f s = %.3f min" % ((end -start), (end - start)/60.0))

	#get all features together
	all_train_features = [train_pragmatic, train_lexical, train_pos, train_sim]
	all_test_features = [test_pragmatic, test_lexical, test_pos, test_sim]

	#print("Feature length %d %d %d %d " % (len(train_pragmatic[0]), len(train_lexical[0]), len(train_pos[0]), len(train_sim[0])) )
	#print("Feature length %d %d %d %d " % (len(test_pragmatic[0]), len(test_lexical[0]), len(test_pos[0]), len(test_sim[0])))

	#print("Feature length %d %d %d %d " % (len(train_pragmatic[1]), len(train_lexical[1]), len(train_pos[1]), len(train_sim[1])) )
	#print("Feature length %d %d %d %d " % (len(test_pragmatic[1]), len(test_lexical[1]), len(test_pos[1]), len(test_sim[1])))

	
	# choose your features options: you can run all on possible combinations of features
	sets_of_features = 4
	#feature_options = list(itertools.product([False, True], repeat = sets_of_features))
	#feature_options = feature_options[1:]

	#or can just choose just the features that you want
	#from left to right ,set true if you want the deatuer to be active
	#[Pragmatic, Lexical-feature, POS_gram, Simlarity]
	feature_options = [[True,True,True, True]]
	flag=0
	for option in feature_options:
		flag+=1
		train_features = [{} for _ in range(len(train_tokens))]
		test_features = [{} for _ in range(len(test_tokens))]
		utils.print_features(option, ['Pragmatic','Lexical-feature','POS_gram','Simlarity'])

		# make a feature selection of features
		for i, o in enumerate(option):
			if o:
				
				for j, example in enumerate(all_train_features[i]):
					train_features[j] = utils.merge_dicts(train_features[j], example)
				for j, example in enumerate(all_test_features[i]):
					test_features[j] = utils.merge_dicts(test_features[j], example)

		# vectorize and scale the features
		x_train, x_test = utils.extract_features_from_dict1(train_features, test_features)
		x_train_scaled = preprocessing.scale(x_train, axis =0)
		x_test_scaled = preprocessing.scale(x_test, axis =0)

		print("Shape of the x train set (%d, %d)" % (len(x_train_scaled), len(x_train_scaled[0])))
		print("Shape of the x test set (%d, %d)" % (len(x_test_scaled), len(x_test_scaled[0])))
	
		#Run the model on the selection of the feature made
		

		start =time.time()
		utils.run_supervised_learning_models(x_train_scaled, y_train, x_test_scaled, y_test)
		end =time.time()
		
		print("Completion time of Linear SVM model: %.3f s = %.3f min" % ((end -start), (end - start)/60.0))

		
	



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
	run_baseline = False
	
	#x_train_features = extract_baseline_features.get_ngram_features(train_tokens[:5], n=1)
	#print((x_train_features))
	
	if run_baseline:
		baseline(train_tokens, train_labels, test_tokens, test_labels)
	else:
		#TODO
		#print("wait!!!")
		ml_model(train_tokens, train_pos, train_labels, test_tokens, test_pos, test_labels)

	


	
