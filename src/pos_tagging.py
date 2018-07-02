from __future__ import unicode_literals
import time
start = (time.time())


from isc_tokenizer import Tokenizer
from isc_tagger import Tagger


tk = Tokenizer(lang='hin')
tagger = Tagger(lang='hin')
print(str(time.time() - start)+" seconds in intializing tokenizer and tagger.\n")
#sequence = tk.tokenize("राम फल खा रहा है| :-)")

tweets = ''
with open('tokens_clean_original_train_hn.txt','r') as filename:
	print("Loading File....")	
	tweets = filename.read().split("\n")

print("Number of sentences loaded .. "+str(len(tweets)))

def get_tag(pos_list):
	tag_list = []
	for word, tag in pos_list:
		tag_list.append(tag)
	return ' '.join(tag_list)

	
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
		print("length of tokenized :"+str(len(sequence)))		
		print("Issue in tagging line"+str(i+1)+'\n')
		break
	pos_tweets.append(get_tag(pos_tw))
	i+=1	
end = time.time()

#print(" Time for pos tagging "+str( end-start )+" seconds for 1 sentences after intializing.")


f = open('pos_train.txt','w')
data = '\n'.join(pos_tweets)
f.write(data)
f.close()

print('Written in output.')

