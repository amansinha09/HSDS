#extract_ml_features.py
import emoji, re, os, time
from gensim.models import LdaModel
from gensim.corpora import MmCorpus, Dictionary
from tqdm import tqdm
import vocab_helper as helper
#import utils
#import pre_processing as dproc
from nltk import ngrams


def build_lda_model(tokens_tags, pos_tags, use_nouns = True, use_verbs = True, use_all = False, num_of_topics = 8, passes=25, verbose = True):
	path = '\\'.join((os.getcwd()).split('\\')[:-1])
	topics_filename = str(num_of_topics) + "topics"

	if use_nouns:
		topics_filename + = "_nouns"
	if use_verbs:
		topics_filename + = "_verbs"
	if use_all:

		topics_filename + = "_all"

	#set the LDA, DIctionary and Corpus filenames
	lda_filename =  path + "\\models\\topics\\lda_"+ topics_filename + ".model"
	dict_filename = path + "\\res\\topic_data\\dict\\dict_" + topics_filename + ".dict"
	corpus_filename = path + "\\res\\topic_data\\corpus\\corpus_" + topics_filename + ".mm"

	#build a topic model if wasn't created yet
	if not os.path.exists(lda_filename)
		# Extract lemmatize document
		docs =[]
		for index in range(len(tokens_tags)):
			tokens = tokens_tags[index].split()
			pos = pos_tags[index].split()
			#docs.append(data_proc.extract_lemmatized_tweets(tokens, pos, use_verbs, use_nouns, use_all))


		#compute dictionary and save it
		dictionary = Dictionary(docs)
		dictionary.filter_extremes(keep_n = 40000)
		dictionary.compactify()
		Dictionary.save(dictionary, dict_filename)

		corpus = [dictionary.doc2bow(d) for d in docs]
		MmCOrpus.serialize(corpus_filename, corpus)

	if verbose:
		print("\nCleaned DOcument:", docs)
		print("\nDictionary:", dictionary)
		print("\nCOrpus is BOW form:", corpus)

	#start training lda model
	start =time.time()
	print("\n BUilding lda topics model....")
	lda_model = LdaModel(corpus=corpus, num_topics = num_of_topics, passes = passes, id2word = dictionary)
	lda_model.save(lda_filename)
	end = time.time()
	print("Completion time for building LDA model: %.3f s = %.3f min" % ((end- start), (end -start)/60.0))

	if verbose:
		print("\nList of words associated with each topics")
		lda_topics_list = [[word for word, prob in topic] for topic_id, topic in lda_topics]
		print([t for t in lda_topics_list])

	#Load the previously saved dictionary 
	dictionary = Dictionary.load(dict_filename)

	#Load the previously saved corpus
	mm_corpus = MmCOrpus(corpus_filename)

	#Load the provious saved LDA model
	lda_model = LdaModel.load(lda_filename)

	# print top 10 for each topic
	if verbose:
		for topic_id in range(num_of_words):
			print("\n atop 10 words for each topics", topic_id)
			print([dictionary[word_id] for (word_id, prob) in lda_model.get_topic_terms(topic_id, topn =10)])

	index=0
	if verbose:
		for doc_topics, word_topics, word_phis in lda_model.get_document_topics(mm_corpus, per_word_topics =True):
			print('Index', index)
			print('Document topics', doc_topics)
			print('Word topics:', word_topics)
			print('Phi values:', word_phis)
			print('--------------------------\n')
			index +=1
	return dictionary, mm_corpus, lda_model

#predict topic of unseen tweet using testing example based lda model built on train set
def get_topic_features_for_unseen_tweet(dictionary, lda_model, tokens_tags, pos_tags, use_nouns=True, use_verbs=True, use_all):

	#extract the lemmatize documents
	docs = data_proc.extract_lemmatized_tweets(tokens_tags, pos_tags, use_verbs, use_nouns, use_all)
	tweet_bow = dictionary.doc2bow(docs)
	topic_prediction = lda_model[tweet_bow]
	topic_features = {}
	if any(isinstance(topic_list, type([])) for topic_list in topic_prediction):
		topic_prediction = topic_prediction[0]
	for topic in topic_prediction:
		topic_features[ 'topic '+str(topic[0])] = topic[1]
	return topic_features

def get_topic_features(corpus, ldamodel, index):
	topic_features = {}
	doc_topics, word_topics, phi_values = ldamodel.get_document_topics(corpus, per_word_topics=True)[index]
	for topic in doc_topics:
		topic_features['topic '+ str(topic[0])] = topic[1]
	return topic_features





