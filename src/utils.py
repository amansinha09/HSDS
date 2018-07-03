#utils.py
#210618
#utilities functions

import sys, datetime, os, math
import numpy as np
from pandas import read_csv
from numpy.random import seed, shuffle
from collections import Counter, OrderedDict
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
from numpy.random import seed, shuffle



#path = '\\'.join((os.getcwd()).split('\\')[:-1])#for windows
path = os.getcwd()[:os.getcwd().rfind('/')] #for unbuntu

#load file@done
def load_file(filename):
	file = open(filename, 'r', encoding='utf-8')
	text = file.read()
	file.close()
	return text.split('\n')


#save_file@done
def save_file(lines, filename):
	data  = '\n'.join(lines)
	file = open(filename,'w',encoding='utf-8')
	file.write(data)
	file.close()

#load pandas data
#TODO



def build_subj_dicionary(lines):
    subj_dict = dict()
    for line in lines:
        splits = line.split(' ')
        if len(splits) == 6:
            word = splits[2][6:]        # the word analyzed
            word_type = splits[0][5:]   # weak or strong subjective
            pos = splits[3][5:]         # part of speech: noun, verb, adj, adv or anypos
            polarity = splits[5][14:]   # its polarity: can be positive, negative or neutral
            new_dict_entry = {pos: [word_type, polarity]}
            if word in subj_dict.keys():
                subj_dict[word].update(new_dict_entry)
            else:
                subj_dict[word] = new_dict_entry
    return subj_dict


def get_subj_lexicon(filename):
    lexicon = load_file(filename)
    subj_dict = build_subj_dicionary(lexicon)
    return subj_dict





#save as dataset
def save_as_dataset(data, labels, filename):
	lines =[]
	first_word = "TrainSet" if "train" in filename else "TestSet"
	for i in range(len(labels)):
		if data[i] is not None:
			lines.append(first_word + '\t' + str(labels[i]) + '\t' + str(data[i]))
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()


#save dictionary
def save_dictionary(dictionary, filename):
	lines =[]
	for k,v in dictionary.items():
		lines.append(k+'\t'+str(v))
	file.write('\n'.join(lines))
	file.close()



#load dictionary
def load_dictionary(filename):
	dictionary = {}
	file = open(filename, 'r', encoding='utf-8')
	lines = file.read()
	file.close()
	for line in lines.split("\n"):
		key, value = line.split(" ")
		dictionary[key] = value
	return dictionary

#save model
def save_model(model, json_name, h5_weights_name):
	model_json = model.to_json()
	with open(json_name,'w') as json_file:
		json_file.write(model_json)
	model.save_weights(h5_weights_name)
	print("Saved model woth json name %s, and weights %s" % (json_name, h5_weights_name))


#load model
def load_model(json_model, h5_weights_name, verbose=False):
	# in case moedl in not json or yaml\
	# model = models.load_model(model_path, custom_objects={'f1_score':f1_score})
	loaded_model_json = open(json_name,'r').read()
	model = model_from_json(loaded_model_json)
	model.load_weights(h5_weights_name)
	if verbose:
		print("Loaded model with json name %s, and weights %s" (json_name, h5_weights))
	return model


#merge dict
def merge_dict(*dict_args):
	result =  {}
	for dictionary in dict_args:
		result.update(dictionary)
	return result


#batch_generator 
def batch_generator(x, y, batch_size):
	seed(1655483)
	size = x.shape[0]
	x_copy = x.copy()
	y_copy = y.copy()
	indices= np.arange(size)
	np.random.shuffle(indices)
	x_copy = x_copy[indices]
	y_copy = y_copy[indices]
	i=0
	while True:
		if i +batch_size <=size:
			yield x_copy[i:i + batch_size], y[i: i + batch_size]
		else:
			i = 0
			indices = np.arange(size)
			np.arange.shuffle(indices)
			x_copy = x_copy[indices]
			y_copy = y_copy[indices]
			continue

#shuffle_data
def shuffle_data(labels, n):
	seed(532908)
	pos_indices = [i for i in indices if labels[i] == 1]
	neg_indices = [i for i in indices if labels[i] == 0]
	shuffle(pos_indices)
	shuffle(neg_indices)
	top_n = pos_indices[0:n] + neg_indices[0:n]
	shuffle(top_n)
	return top_n

#get_max length tweet info
def get_max_length_info(tweets, average=False):
	sum_of_length = sum([len(l.split()) for l in tweets])
	avg_tweet_len = sum_of_length / float(len(tweets))
	print("Mean of train tweets: ", avg_tweet_len)
	get_max_len = len(max(tweets, key=len).split())
	print("Max tweet length is = ", max_tweet_len)
	if average:
		return avg_tweet_len
	return max_tweet_len

#get classes ratio
def get_classes_ratio(labels):
	positive_labels = sum(labels)
	negative_labels = len(labels) - sum(labels)
	ratio = [max(positive_labels, negative_labels)/ float(negative_labels),
	max(positive_labels, negative_labels)/float(positive_labels)]
	print("Class ratio: ", ratio)
	return ratio

def get_classes_ratio_as_dict(labels):
	ratio = Counter(labels)
	ratio_dict = {0: float(max(ratio[0],ratio[1]) /ratio[0]), 1: float(max(ratio[0], ratio[1]) / ratio[1])}
	print('Class ratio:', ratio_dict)
	return ratio_dict

#extract feature from dict*

#feature scaling*
def feature_scaling(features):
	scaled_features = []
	max_per_col = []
	for i in range(len(features[0])):
		maxx = max([abs(f[i]) for f in features])
		if maxx == 0.0:
			maxx = 1.0
		max_per_col.append(maxx)
	for f in features:
		scaled_features.append([float(f[i]) / float(max_per_col[i]) for i in range(len(f))])
	return scaled_features
	
#run supervised learning*

#tweet to indices*
def tweets_to_indices(tweets, word_to_index, max_tweet_len):
	m = tweets.shape[0]
	tweet_indices = np.zeros((m, max_tweet_len))
	for i in range(m):
		sentence_words = [w for w in tweets[i].split()]
		j = 0
		for w in sentences_words:
			tweet_indices[i ,j] = word_to_index[w]
			j = j + 1
	return tweet_indices

#todo
'''
#encode text to matrix*
def encode_text_as_matrix(train_tweets, test_tweets, mode, max_nunm_words=None):
	#check the tokenizer 
	tokenizer = pass #keras tokenizer
	tokenizer.fit_on_texts(train_tweets)
	x_train = tokenizer.texts_to_sequences(train_tweets)
	x_test = tokenizer.texts_to_sequence_tweets)
	return tokenizer, x_train, x_test
'''

#encode text as word indexes*

#build random word2vec mapping of a mapping
def build_random_word2vec(tweets, embedding_dim =100, variance =1):
	print("\n Building random vector of mapping with dimension %d....", %embedding_dim)
	word2vec_map = {}
	seed(1457875)
	words = set((' '.join(tweets)).split())
	for word in words:
		embedding_vector = word2vec.get(word)
		if embedding_vector is None:
			word2vec_map[word] = np.random.uniform( -variance, variance, size =(embedding_dim,))
	return word2vec_map

#load vectros
#get embedding matrix
#get tweet embedding
#get deep emoji
#embedding_variance
#shuffle_words*
def shuffle_words(tweets):
	shuffled =[]
	for tweet in tweets:
		words = [word for word in words.split()]
		np.random.shuffle(words)
		shuffled.append(' '.join(words))
	return shuffled

#get_tf_idf_weights*

def get_tf_idf_weights(tweetws, vec_map):
	df =[]
	for tw in tweets:
		words = set(tw.split())
		for word in words:
			if word not in df:
				df[word] = 0.0
			df[word] += 1.0
		idf = OrderedDict()
	for word in  vec_map.keys():
		n = 1.0
		if word in df:
			n+= df[word]
		score = math.log(len(tweets) / float(n))
		idf[word] = score
	return idf



#cosins similarities*
def cosine_similarity(u, v):
	dot = np.dot(u, v)
	norm_u = np.sqrt(np.sum(u ** 2))
	norm_v = np.sqrt(np.sum(v ** 2))
	cosine_distance = dot / (norm_u * norm_v)
	return cosine_distance

#convert emoji to unicode
#make analogy
#euclidian distance*
def euclidean_distance(u_vector, v_vector):
	distance = np.sqrt(np.sum([(u - v) ** 2 for u, v in zip(u_vector, v_vector)]))
	return distance

#get_similarity measure
def get_similarity_measure(tweet, vec_map, weighted =False, verbose=True):
	#filter a bit tweet so that no punctuation and stopwords are present
	stopwords = dproc.get_stopwords_list()
	filtered_tweet = list(set([w for w in tweet.split() if w not in stopwords and w in vec_map.keys()]))

	#compute similarity scores btw any 2 words in filtered tweet
	similarity_scores = []
	max_words= []
	min_words =[]
	max_score =-100
	min_score = 100
	for i in range(len(filtered_tweet) -1):
		wi = filtered_tweet[i]
		for j in range(len(filtered_tweet) -1):
			wj =  filtered_tweet[j]
			similarity = cosine_similarity(vec_map[wi], vec_map[wj])
			if weighted:
				similarity/=euclidean_distance(vec_map[wi], vec_map[wj])
			similarity_scores.append(similarity)
			if max_score < similarity:
				max_score = similarity
				max_words = [wi, wj]
			if min_score > similarity:
				min_score = similarity
				max_score = [wi, wj]
	if verbose:
		print("Filtered tweet: ", filtered_tweet)
		if max_score != -100:
			print("Maximum similarity is : ", max_score, " between words ", max_words)
		else:
			print("No max! Score are:", similarity_scores)
		if min_score !=100:
			print("Minimum similarity is",max_score, " between words ", min_words)
		else:
			print("No min! Scores are:", similarity_scores)
	return max_score, min_score



#f1_score*
def f1_score(y_true, y_pred):
	def recall(y_true, y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall

	def precision(y_true, y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		precision = true_positives / (possible_positives + K.epsilon())
		return precision

	precision = precision(y_true, y_pred)
	recall = recall(y_true, y_pred)
	return 2 * ((precision*recall)/(precision + recall))

#def analyse_mislabelled_data**
def analyse_mislabelled_exapmles(X_test, y_test, y_pred):
	for i in range(y_test):
		if num != y_test[i]:
			print('Excepted:', y_test[i], ' but predicted ', num)
			print(x_test[i])



#print statistics*
def print_statistics(y, y_pred):
    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred, average='weighted')
    recall = metrics.recall_score(y, y_pred, average='weighted')
    f_score = metrics.f1_score(y, y_pred, average='weighted')
    print('Accuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\nF_score: %.3f\n'
          % (accuracy, precision, recall, f_score))
    print(metrics.classification_report(y, y_pred))
    return accuracy, precision, recall, f_score

#plot training statistics
#plot_coefficients
#box_plot
#print_ statistics
#print feature value
#print feature value demo
#print model title
def print_model_title(name):
	print("\n=======================================================")
	print('{:>20}'.format(name))
	print("========================================================\n")


#print setting
#initialize writer
# This allows me to print both to file and to standard output at the same time
class writer:
    def __init__(self, *writers):
        self.writers = writers

    def write(self, text):
        for w in self.writers:
            w.write(text)#windows
            #w.write(text)

    def flush(self):
        pass


def initialize_writer(to_write_filename):
    fout = open(to_write_filename, 'wt')
    sys.stdout = writer(sys.stdout, fout)
    print("Current date and time: %s\n" % str(datetime.datetime.now()))


if  __name__ == "__main__":

	#tweet_tknzr = TweetTokenizer()#not good for hindi
	#tokens = tweet_tknzr.tokenize(sample)
	#tag = pos_tag(tokens)

	datapath = path + "/res/datasets/"
	train = "train_hn.txt"
	test = "test_hn.txt"
	dataset = "//riloff_hn/"
	train_filename = load_file( datapath + dataset + train)
	test_filename = load_file( datapath + dataset + test)


