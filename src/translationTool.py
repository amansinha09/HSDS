# -*- coding: utf-8 -*- 

#from googletrans import Translator
import time,  os

a = (os.getcwd())
path = '\\'.join(a.split('\\')[:-1])

#list of sentences to be translated
src_en = ['Hey there, how are you','This is my car']

to_write_filename = path + '\\stats\\bag_of_words_analysis.txt'
train_tweets_filename = "train.txt"
test_tweets_filename = "test.txt"
#check for which dataset to perform operations datasets = ["hercig","ghosh","sarcasmdetection","riloff"]
datasets = ["hercig","riloff","ghosh"]
datapath = path + "\\res\\datasets\\"

def load_file(filename):
    #file = open(filename, 'r')
    file = open(filename,'r',encoding="utf8") #windows
    text = file.read()
    file.close()
    return text.split("\n")


def save_file(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w',encoding="utf-8")#windows
    file.write(data)
    file.close()

def countDataset():
	for dataset in datasets:
		print("Dataset: [%s]" %dataset)
		lines = utils.load_file(datapath + dataset +"\\tokens_" + train_tweets_filename)
		print("Number of sentences for training: %d" %(len(lines)))

		lines = utils.load_file(datapath + dataset +"\\tokens_" + test_tweets_filename)
		print("Number of sentences for testing: %d\n" %(len(lines)))

#countDataset()
#[dataset] 	<train, test>
#[hercig] 	<14340, 3585>
#[riloff] 	<1368,	588>
#[ghosh] 	<51189, 3742>
#[sd] 		<,	>




'''
lang ='hi'
dataset = 'riloff'
datapath = datapath + dataset +"_hn\\" 
print("Loading data....")
lines = utils.load_file('E:\\nlp+\\SDS-master\\res\\datasets\\hercig_hn\\train.txt')
#(datapath+train_tweets_filename)
print("%d sentences loaded. "%len(lines))
#print(datapath)



T = Translator()

start = time.time()
print("Translation started....")
i=1
tr=[]
errsen =[]# sentences that couldnt be translatd format=<idx /tab sentences>

for line in lines:
	try:
		translation = T.translate(text =line, src='en',dest='hi')
	except:
		errsen.append(str(i)+"\t"+line)
		print("error occured at %dth sentence" %i)
	i=i+1

	tr.append((translation.text))

end = time.time()
	
utils.save_file(tr,'E:\\nlp+\\SDS-master\\res\\datasets\\hercig_hn\\hi_train.txt')
utils.save_file(errsen,'E:\\nlp+\\SDS-master\\res\\datasets\\hercig_hn\\err_train.txt')

#E:\nlp+\SDS-master\res\datasets\hercig_hn\train_hn.txt
#E:\nlp+\SDS-master\res\datasets\riloff_hn\train.txtriloff\tokens_train_hn.txt
#print(datapath + dataset +"\\" + train_tweets_filename[:-4]+"_hn.txt")

print("Translation saved.")
print("\nTime taken per translation = %d sec \n" %((end-start)/(int(len(lines))) ))
'''

''''''

def upd():
	lines=load_file('E:\\nlp+\\SDS-master\\res\\datasets\\riloff_hn\\test.txt')
	start = int(input("where to start:"))
	updatedLines = lines[start:]
	print("New length :%s" %len(updatedLines))
	save_file(updatedLines,'E:\\nlp+\\hsds-master\\res\\datasets\\riloff_hn\\eng.txt')
	print("File updated.")
upd()
	


	
