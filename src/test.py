from __future__ import unicode_literals
import time, sys, os
from isc_tokenizer import Tokenizer
from isc_tagger import Tagger

start = time.time()
tk = Tokenizer(lang='hin')
tagger = Tagger(lang='hin')
print(str(time.time() - start)+" seconds. Tokenizer and Tagger initialized.\n")

#testing tokenizing
#sequence = tk.tokenize("राम फल खा रहा है| :-)")
#sys.stdout = open("isc_test.txt", "a", encoding='utf-8')
#print(tagger.tag(sequence))
#path = '\\'.join((os.getcwd()).split('\\')[:-1]) #for windows

path = os.getcwd()[:os.getcwd().rfind('/')] #for unbuntu
tokens_path = path + "/res/tokens/"
tweets = ''

#Loading training file
with open(tokens_path + 'tokens_DIRTY_original_train_hn.txt','r') as filename:
	print("Loading File....")	
	tweets = filename.read().split("\n")

print("Number of sentences loaded .. "+str(len(tweets)))
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


def data_for_word2vec(tweets, tokenizer):
	#remove symbols






















#Writing output in text file
f = open(path+'/res/pos/' + 'DIRTY_pos_train.txt','w')
data = '\n'.join(pos_tweets)
f.write(data)
f.close()

print('Written in output.')

