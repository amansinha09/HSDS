from __future__ import unicode_literals
import time, sys, os
from isc_tokenizer import Tokenizer
from isc_tagger import Tagger
import pre_processing as dproc
import utils
import extract_ml_features as emf
import emoji

#start = time.time()
def hin_tool():
	tk = Tokenizer(lang='hin')
	tagger = Tagger(lang='hin')
	return tk, tagger
#print(str(time.time() - start)+" seconds. Tokenizer and Tagger initialized.\n")

#testing tokenizing
#sequence = tk.tokenize("राम फल खा रहा है| :-)")
#sys.stdout = open("isc_test.txt", "a", encoding='utf-8')
#print(tagger.tag(sequence))
#path = '\\'.join((os.getcwd()).split('\\')[:-1]) #for windows

path = os.getcwd()[:os.getcwd().rfind('/')] #for unbuntu
tokens_path = path + "/res/tokens/"
tweets = ''
'''
#Loading training file
with open(tokens_path + 'tokens_DIRTY_original_train_hn.txt','r') as filename:
	print("Loading File....")	
	tweets = filename.read().split("\n")

print("Number of sentences loaded .. "+str(len(tweets)))
'''																																																																																																																																																																																											
#separating [(word_i, tag_i)] => [(tag_i)]
def get_tag(pos_list):
	tag_list = []
	for word, tag in pos_list:
		tag_list.append(tag)
	return ' '.join(tag_list)
###############################################
#genrating pos
'''
start =time.time()
pos_tweets = []
i=0
for tweet in tweets:
	#print("Tokenizing tweet.." + str(i+1))	
	try:
		sequence = tk.tokenize(tweet)
	except:
		print("Issue in tokenizing line "+ str(i+1)+'\n')
		break
	#print("Tagging tweet.." + str(i+1))
	try:
		if len(sequence)==0:
			pos_tweets.append('')
			i+=1
			continue		
		pos_tw = tagger.tag(sequence)
	except:
		#problem when tokenized output is empty so checked length
		print("length of tokenized :"+str(len(sequence)))		
		print("Issue in tagging line"+str(i+1)+'\n')
		break
	pos_tweets.append(get_tag(pos_tw))
	i+=1	
end = time.time()
'''
################################################
sample = 'मैं लगातार ट्विटर पर आर्सेनल के बारे में ट्वीट्स देखता हूं। दुनिया को अपडेट करने के लिए धन्यवाद @उपयोगकर्ता & @उपयोगकर्ता शॉनक्स। #'


def data_for_word2vec(tweets):
	#remove symbols
	start = time.time()
	tokenizer , tagger = hin_tool()
	print(str(time.time() - start)+" seconds. Tokenizer and Tagger initialized.\n")
	#Tokenizer(lang='hin')
	clean_tweets = dproc.get_clean_tweet(tweets)
	#print("Clean tweet:", clean_tweets)
	#tagger = Tagger(lang='hin')
	dwv = []
	for tw in clean_tweets:
		tokens = tokenizer.tokenize(tw)
		#print(len(tokens))
		if(len(tokens)<1):
			dwv.append('')
			continue
		tags = tagger.tag(tokens)
		valid_tokens = []
		for p in tags:
			if p[1] != 'SYM' and p[0] !='#' and p[0][0]!='@' and p[0] not in emoji.EMOJI_UNICODE:
				valid_tokens.append(p[0])
		cl_tw = ' '.join(valid_tokens)
		dwv.append(cl_tw)
	return (dwv)

#sys.stdout = open("toutput.txt", "a", encoding='utf-8')
# load tweets
#d = data_for_word2vec(tweets)
#print(sample,"\n===================================\n")
#print(d)

#d = 
#print(type(d[0]))

dataset4wv = 'w2v'
#========================================================
def use_w2v():
	tweets =  utils.load_file(tokens_path + 'w2v_train.txt')
	print("tweets loaded...")
	d = data_for_word2vec(tweets)
	print("cleaning done....")
	w2v = utils.build_random_word2vec(d)
	print("w2v ready...")
	return w2v

#feature = emf.get_similarity_scores(sample, w2v)
#print((feature))

#========================================================

'''
#Writing output in text file
f = open(path+'/res/tokens/' + dataset4wv +'_train.txt','w')
data = '\n'.join(d)
f.write(data)
f.close()

print('Written in output.')
'''

#count vocabulary ==============================(train or test)=====================
if __name__ == '__main__':
	tweets1 = open(tokens_path + 'w2v_train.txt','r').read().split('\n')
	tweets2 = open(tokens_path + 'w2v_test.txt','r').read().split('\n')
	print(len(tweets1 + tweets2))
	print(len(tweets2))
#tweets1 =set(' '.join(tweets1).split(' '))
#tweets2 = set(' '.join(tweets2).split(' '))
#tweet = tweets2.union(tweets1)

#print(len(tweet))

#print("chill hai!!")
