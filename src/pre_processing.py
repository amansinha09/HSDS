#preprocessing.py
#210618
#pre-processing of data
#each sentence is hindi translation of english/ sometimes other language tweets


import re, os, itertools, string
from collections import Counter
import numpy as np
import emoji
#import utils
#from vocab_helpers import *

path = '\\'.join((os.getcwd()).split('\\')[:-1])
dict_filename = "word_list.txt"
#word_filename =  "word_list_freq.txt"

#build subjective dictionary
#build subjective lexicon
#get subj lexicon
#get emoji dictionary
def get_emoji_dictionary():
	emojis = utils.load_file(path + "\\res\\emoji\\emoji_list.txt")
	emoji_dict = {}
	for line in emojis:
		line = line.split(" ",1)
		emoji = line[0]
		description = line[1]
		emoji_dict[emoji] = description
	return emoji_dict


#build emoji sentiment dictionary
def build_emoji_sentiment_dictionary():
	new_emoji_sentiment_filename = path + "\\res\\emoji\\emoji_sentiment_dictionary.txt"
	if not os.path.exists(new_emoji_sentiment_filename):
		filename = path + "\\res\\emoji\\emoji_sentiment_raw.txt"
		emojis = utils.load_file(filename)[1:]
		lines =[]
		for linein emojis:
			line = line.split(",")
			emoji = line[0]
			occurences = line[2]
			negative = 
			neutral =
			positive = 
			description = line[7]
			lines.append(str(emoji)+"\t"+str(negative)+"\t"+str(neutral)+"\t"+str(postive)+"\t"+description.lower())
			utils.save(lines, new_emoji_sentiment_filename)
	emoji_sentiment_data =utils.load_file(new_emoji_sentiment_filename)
	emoji_sentiment_dict = {}
	for line in emoji_sentiment_data:
		line = line.split("\t")
		emoji_sentiment_dict[line[0]] = [line [1], line[2], line[3], line[4]]
	return emoji_sentiment_dict

#extract emoji

def extract_emoji(tweets):
	emojis =[]
	for tw in tweets:
		tw_emojis =[]
		for word in tw:
			chars = list(word)
			for ch in chars:
				if ch in emoji.UNICODE_EMOJI:
					tw_emojis.append(ch)
		emojis.append(' '.join(tw_emojis))
	return emojis

#replace contraction
#correct spelling
#reduce lengthening
#process emoji
#grammar cleaning
#get stopwordlist
def get_stopwords_list(filename="stopwords.txt"):
	stopwords = utils.load_file(path + "\\res\\"+filename)
	return stopwords

#build vocabulary
def build_vocabulary(vocab_filename, lines, minimum_occurrence=1):
	if not os.path.exists(vocab_filename):
		stopwords = get_stopwords_list(filename="stopwords.txt")#check
		print("Building vocabulary....")
		vocabulary = Counter()
		for line in lines:
			vocabulary.update([l for l in line.split() if l not in stopwords])
		print("The top 10 most common words:", vocabulary.most_common(10))
		vocabulary = {key: vocabulary[key] for key in vocabulary if vocabulary[key] >= minimum_occurrence}
		utils.save_file(vocabulary.keys(), vocab_filename)
		print("vocabulary saved to file \"%s\""% vocab_filename)
	vocabulary = set(utils.load_file(vocab_filename))
	print("Loaded vocabulary of size", len(vocabulary))
	return vocabulary



#build vocabulary for dnn
#vocabulary filtering
#extract lemmatization tweets
#filter_based vocab
#ulterior clean
#get_tag_for_each tweet
#cmu_prob
#camel case split
def camel_case_split(term):
	term = re.sub(r'[0-9]',r' \l', term)
	term = re.sub(r'(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|0th)', r'\1 ', term)
	splits = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', term)
	return [s.group(0) for s in splits]

#split_hashtag to all possibilities
#split hashtag
#split hashtag long version
#split hashtag2
#clean tweet
def clean_tweet(tweet, word_list, split_hashtag_method, replace_user_mentions=True, remove_hastags=True, remove_emojis=True):
	tweet = re.sub(,tweet)
	tweet = re.sub(, tweet)
	valid_tokens = []
	for word in tokens:
		#remove sarcasm hashtag
		if word.startswith("#sarca"):
			continue
		#remove urls
		if 'http' in word:
			continue
		#remove user mentions
		if replace_user_mentions:
			pass #todo
		#remove hashtags
		if word.startswith("#"):
			if remove_hastags:
				continue
			splits = split_hashtag_method(word[1:], word_list)
			valid_tokens.extend([split for split in splits])
			continue
		#remove emojis
		if remove_emojis and word in emoji.UNICODE_EMOJI:
			continue
		valid_tokens.append(word)
	return ' '.join(valid_tokens)



#processtweet
def process_tweets(tweets, word_list, split_hashtag_method):
	clean_tweets =[]
	for tweet in tweets:
		clean_tw = clean_tweets(tweet, word_list, split_hashtag_method)
		clean_tweets.append(clean_tw)
	return clean_tweets



#process set
def process_set(dataset_filename, vocab_filename, word_list, min_occ=10):
	data, labels = utils.load_panda_data_panda(dataset_filename)
	tweets = process_tweets(data, word_list, split_hashtag)
	vocabulary = build_vocabulary(twe, vocab_filename, minimum_occurrence=min_occ)
	filtered_tweets =[]


#initial clean
#check if emoji
def check_if_emoji(word, emoji_dict):
    emojis = list(word)
    for em in emojis:
        if em in emoji_dict.keys() or em in emoji.UNICODE_EMOJI:
            return True
    return False

#strict emoji

#strict clean
#get strict data
#get clean data
#get filtered data
#get grammatical data
#get clean dl data
#get dataset
def get_dataset(dataset):
	data_path = path + "\\res\\datasets\\" + dataset + "\\"
	train_tweets = 
	test_tweets = 
	train_pos = 
	test_pos =
	train_labels = 
	test_labels =
	print("Size of the train set:", len(train_labels))
	print("Size of the test set", len(train_labels))
	return train_tweets, train_pos, train_labels, test_tweets, test_pos, test_labels




if __name__ == '__main__':

	train_filename = "train.txt"
	test_filename = "test.txt"

	# For a superficial clean 
	clean_train, clean_test = get_clean_data(train_filename, test_filename, word_filename)

