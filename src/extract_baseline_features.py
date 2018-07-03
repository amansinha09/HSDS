#extract_baseline_features.py

from __future__ import unicode_literals
import time, sys, ast
from isc_tokenizer import Tokenizer
from isc_tagger import Tagger
from nltk import ngrams, pos_tag
import numpy as np
from collections import Counter#collections in python 3
import vocab_helpers as helper
#import pre_processing as dproc


sample_ = 'Thomas, and @tom you go to bed right now or I will kill you! ;-) #joke'
sample =['मेरे दांतों को इतनी परेशान करो ! ! ! ओह अच्छी तरह से सीधे दांतों के लिए कुछ भी।', 'मैं उम्मीद कर रहा था कि इस हफ्ते सिर्फ थोड़ी सी चीज प्रशंसक को मार सकती है।']#,'मैं इस साल फुटबॉल टीम के लिए खेलूँगा। @फिफा :)']

def count_apparitions(tokens, list_to_count_from):
	total_count =0.0
	for affirmative in list_to_count_from:
		total_count += tokens.count(affirmative)
	return total_count

def get_features1(tweets, subj_dict):

	print("Getting features type 1 ... : [p_verb, n_verb, p_noun, n_noun, punctation, negations]")
	features =[]
	tknzr = Tokenizer(lang = 'hin')
	tagger = Tagger(lang='hin')
	#take positive and negative noun/verb phrases
	for tweet in tweets:
		feature_list = [0.0]*6
		tokens = tknzr.tokenize(tweet)
		pos = tagger.tag(tokens)
		#print("=>",pos,'\n')
		pos = [p for p  in pos if 'V' in p[1] or 'NN' in p[1]]
		#print("==>",pos,'\n')
		for p in pos:
			word = p[0]
			if 'V' in p[1] and word in subj_dict:
				if 'verb' in subj_dict[word]:
					if 'positive' in subj_dict[word]['verb']:
						feature_list[0]+=1.0
					if 'negative' in subj_dict[word]['verb']:
						feature_list[1]+=1.0
				elif 'anypos' in subj_dict[word]:
					if 'positive' in subj_dict[word]['anypos']:
						feature_list[0]+=1.0
					if 'negative' in subj_dict[word]['anypos']:
						feature_list[1]+=1.0
			if 'NN' in p[1] in pos and word in subj_dict:
				if  'noun' in subj_dict[word]:
					if 'positive' in subj_dict[word]['noun']:
						feature_list[2]+=1.0
					if 'negative' in subj_dict[word]['noun']:
						feature_list[3]+=1.0
				elif 'anypos' in subj_dict[word]:
					if 'positive' in subj_dict[word]['anypos']:
						feature_list[2]+=1.0
					if 'negative' in subj_dict[word]['anypos']:
						feature_list[3]+=1.0
		#derive feature from punctuations
		feature_list[4]+= count_apparitions(tokens, helper.punctuation)
		#derive number of strong negations words
		feature_list[5]+= count_apparitions(tokens, helper.strong_negations)

		features.append(feature_list)
	print("Done")
	return features


def get_features2(tweets, subj_dict):
	print("Getting features 2....[positive, neagtive, punct, strong_neg, strong_affirm]")
	features =[]
	tknzr = Tokenizer(lang ='hin')
	for tweet in tweets:
		feature_list = [0.0]*5
		tokens = tknzr.tokenize(tweet)
		# take number of positive and negativ word as featuer
		for word in tokens:
			if word in subj_dict:
				dictlist = []
				for _ in subj_dict[word]:
					dictlist.extend(subj_dict[word][_])
					if len(dictlist)>1:
						value = 0.5
					else:
						value = 1.0#check???
				if 'positive' in dictlist:
					feature_list[0] += value
				elif 'negative' in dictlist:
					feature_list[1] += value
		#take the report of positive to negative as feature
		if feature_list[0] != 0.0 and feature_list[1] != 0.0:
			feature_list[2] = feature_list[0] / feature_list[1]
		#derive feature from punctuation
		feature_list[2] += count_apparitions(tokens, helper.punctuation)
		#take strong negation as a feature
		feature_list[3] += count_apparitions(tokens, helper.strong_negations)
		#take_strong aFFIRMATIVES as feature
		feature_list[4] += count_apparitions(tokens, helper.strong_affirmatives)
		features.append(feature_list)
	print("Done.")
	return features


#def feature3 


def  get_ngrams_list(tknzr, text, n):
	tokens = tknzr.tokenize(text)
	tokens = [t for t in tokens if not t.startswith('#')]
	tokens = [t for t in tokens if not t.startswith("@")]
	ngram_list = [gram for gram in ngrams(tokens, n)]
	return ngram_list


def get_ngrams(tweets, n):
	unigrams = Counter()
	bigrams = Counter()
	trigrams = Counter()
	#regrexp_tknzr = RegexpTokenizer(r'\w+')
	tweet_tknzr = Tokenizer(lang='hin')
	for tweet in tweets:
		#tweet = tweet.lower()

		unigram_list = get_ngrams_list(tweet_tknzr, tweet, 1)
		unigrams.update(unigram_list)

		if n>1:
			bigram_list = get_ngrams_list(tweet_tknzr, tweet, 2)
			bigrams.update(bigram_list)
			if n>2:
				trigram_list  =  get_ngrams_list(tweet_tknzr, tweet, 3)
				trigrams.update(trigram_list)
	min_occurence = 2
	unigram_tokens = [k for k,c in unigrams.items() if c >= min_occurence]
	bigram_tokens = trigram_tokens = []
	if n > 1:
		bigram_tokens = [k for k, c in bigrams.items() if c>= min_occurence]
	if n > 2:
		trigram_tokens = [k for k, c in trigrams.items() if c >= min_occurence]
	return unigram_tokens, bigram_tokens, trigram_tokens


def create_ngram_mapping(unigrams, bigrams, trigrams):
	ngram_map = dict()
	all_ngrams = unigrams
	all_ngrams.extend(bigrams)
	all_ngrams.extend(trigrams)
	for i in range(0, len(all_ngrams)):
		ngram_map[all_ngrams[i]]=i
	return ngram_map

def get_ngram_features_from_map(tweets, ngram_map, n):
    #regexp_tknzr = RegexpTokenizer(r'\w+')
    tweet_tknzr = Tokenizer(lang='hin')
    features = []
    for tweet in tweets:
        feature_list = [0] * np.zeros(len(ngram_map))
        tweet = tweet.lower()
        ngram_list = get_ngrams_list(tweet_tknzr, tweet, 1)
        if n > 1:
            ngram_list += get_ngrams_list(tweet_tknzr, tweet, 2)
        if n > 2:
            ngram_list += get_ngrams_list(tweet_tknzr, tweet, 3)
        for gram in ngram_list:
            if gram in ngram_map:
                feature_list[ngram_map[gram]] += 1.0
        features.append(feature_list)
    return features

def get_ngram_features(tweets, n):
    print("Getting n-gram features...")
    unigrams = []
    bigrams = []
    trigrams = []
    if n == 1:
        unigrams, _, _ = get_ngrams(tweets, n)
    if n == 2:
        unigrams, bigrams, _ = get_ngrams(tweets, n)
    if n == 3:
        unigrams, bigrams, trigrams = get_ngrams(tweets, n)
    ngram_map = create_ngram_mapping(unigrams, bigrams, trigrams)
    features = get_ngram_features_from_map(tweets, ngram_map, n)
    print("Done.")
    return ngram_map, features


def p(cmd):
	print(cmd)

#thing incorporated from utils
def load_file(filename):
	file = open(filename, 'r', encoding = 'utf-8')
	text = file.read()
	file.close()
	return text.split('\n')

def get_subj_lexicon(filename):
	lexicon = load_file(filename)
	subj_dict = build_subj_dictionary(lexicon)
	return subj_dict

def build_subj_dictionary(lines):
	subj_dict={}
	for line in lines:
		splits = line.split('\t',1)
		key = splits[0]
		value = ast.literal_eval(splits[1])
		subj_dict[key] = value
		#print(type(key), type(value))
	return subj_dict

#----------------------------
#def 
if __name__ == '__main__':

	#tweet_tknzr = Tokenizer(lang='hin')
	#nltk.TweetTOkenizer is not good for hindi
	#tagger = Tagger(lang='hin')

	#load lexicon


	#tokens = tweet_tknzr.tokenize(sample)
	#tag = pos_tag(tokens)
	#print("Tagger and Tokenizer initialized.")
	#tags = tagger.tag(tokens)
	sd = get_subj_lexicon('hindi_lexicon.tff')

	sys.stdout = open("output.txt", "a", encoding='utf-8')
	#print(sd)

	#pos = [p for p in tag if 'VB' in p[1] or 'NN' in p[1]]
	#print(get_ngrams_list(tweet_tknzr, sample, 3))#get_ngrams_list(tweet_tknzr, (sample),3))
	#print(pos)
	
	
	
	#for testing
	u,b,t = get_ngrams(sample, 3)
	ngram_map = create_ngram_mapping(u,b,t) #it is a dictionry {tuple:index}
	f = get_ngram_features_from_map(sample, ngram_map, 3)

	#ngm, f = get_ngram_features(sample, 3)
	p("feature 4:")
	p(f)#get_ngram_features_from_map(sample, 2))
	#p("features")
	#p(f)
	p("#######################################END-OF-OUTPUT###############################################")

	#p(len(sample[0]))#not tokenized sogiving number of chars
	#p(len(f[0]))
	#for t,n in (ngram_map):
	#	print(t+":"+n)
	#print("Bigrams :\n", (t))

