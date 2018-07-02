from __future__ import unicode_literals
from isc_tokenizer import Tokenizer
#extract_baseline_features.py
import sys
from nltk import ngrams#, pos_tag
import numpy as np
from collections import Counter#collecetions in python

sample = 'Thomas, and @tom you go to bed right now or I will kill you! ;-) #joke'

#'@user इस साल अपने पहले खेल! ! ! ! ! इंडी के बारे में क्या (केवल विनियमन)'
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
	#tweet_tknzr = TweetTokenizer()
	tweet_tknzr = Tokenizer(lang='hin')
	for tweet in tweets:
		tweet = tweet.lower()

		unigram_list = get_ngrams_list(tweet_tknzr, tweet, 1)
		unigrams.update(unigram_list)

		if n>1:
			bigram_list = get_ngrams_list(tweet_tknzr, tweet, 2)
			bigrams.update(bigram_list)
			if n>2:
				trigram_list  =  get_ngrams_list(tweet_tknzr, tweet, 3)
				trigrams.update(trigram_list)
	min_occurence = 1
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
    regexp_tknzr = RegexpTokenizer(r'\w+')
    tweet_tknzr = TweetTokenizer()
    features = []
    for tweet in tweets:
        feature_list = [0] * np.zeros(len(ngram_map))
        tweet = tweet.lower()
        ngram_list = get_ngrams_list(tweet_tknzr, tweet, 1)
        if n > 1:
            ngram_list += get_ngrams_list(regexp_tknzr, tweet, 2)
        if n > 2:
            ngram_list += get_ngrams_list(regexp_tknzr, tweet, 3)
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

if __name__ == '__main__':

	tweet_tknzr = TweetTokenizer()#not good for hindi
	tokens = tweet_tknzr.tokenize(sample)
	tag = pos_tag(tokens)
	sys.stdout = open("output.txt", "a", encoding='utf-8')
	pos = [p for p in tag if 'VB' in p[1] or 'NN' in p[1]]
	print(tag)#get_ngrams_list(tweet_tknzr, (sample),3))
	print(pos)
	
	
	'''
	#for testing
	u,b,t = get_ngrams(sample, 3)
	ngram_map = create_ngram_mapping(u,b,t) #it is a dictionry {tuple:index}
	#f = get_ngram_features_from_map(sample, ngram_map, 3)

	ngm, f = get_ngram_features(sample, 3)
	p("feautres map:")
	p(ngm)
	p("feaures")
	p(f)
	'''

	#p(len(sample[0]))#not tokenized sogiving number of chars
	#p(len(f[0]))
	#for t,n in (ngram_map):
	#	print(t+":"+n)
	#print("Bigrams :\n", (t))

