from __future__ import unicode_literals
import utils, os, re, sys, ast 


#build hindi subjective lexicon
'''
format: type: dict()
		word : {(pos_tag : positive/negative/neutral), ..}

'''
path = '\\'.join((os.getcwd()).split('\\')[:-1])
#pos_dic ={'a':'adj','n':'noun','v':'verb','r':'adv','u':'anypos'}

#print("Current Path:\n"+ path)

#[HN_AMBI, HN_POS, HN_NEG, HN_NEU]

sys.stdout = open("output.txt", "a", encoding='utf-8')
pos_c = utils.load_file(path +"\\res\\pos_lexi.tff")
neg_c = utils.load_file(path +"\\res\\neg_lexi.tff")
neu_c = utils.load_file(path +"\\res\\neu_lexi.tff")
amb_c = utils.load_file(path +"\\res\\ambi_lexi.tff")

lexi_c = [pos_c, neg_c, neu_c, amb_c]

def merge_dict(dict2, dict_args):
    result  = dict2
    for dictionary in dict_args:
        #print(type(dictionary))
        for key in dictionary:
        #.key(), dictionary.value()
            if key not in result:
                #result.update({key:dictionary[key]})
                result[key] = dictionary[key]
            else:
                result[key]+=dictionary[key]
    return result

#i=0
def build_dict(dictionary):
	subj_dict ={}
	i=0
	for line in dictionary:
		splits = line.split('\t',1)
		#print(len(splits))
		if len(splits)!=2:
			#print("Issue at line "+ str(i+1))
			break

		key = splits[0]
		val = ast.literal_eval(splits[1])
		subj_dict[key] = val
		i+=1
	print("Length of the dictionary"+str(len(subj_dict)))
	return subj_dict




def build_lexicon():
	sd = build_dict(pos_c)
	sd1= build_dict(neg_c)
	sd2= build_dict(amb_c)
	sd3 =build_dict(neu_c)

	dict_args  = [sd1 ,sd2, sd3]
	

	result = sd
	print("length of initial lexicon: "+ str(len(result)))
	#print(dict_args[0][:5])
	
	for dictionary in dict_args:
		#print(type(dictionary))
		i=0
		n =0
		f=0
		for key in dictionary:
			if key not in result:
				n+=1
				result.update({key:dictionary[key]})
			else:
				f+=1
				#print(type(key))
				#print(type(result[key]))
				#print(type(dictionary[key]))
				result[key] = merge_dict(result[key], [dictionary[key]])
				#print(key +" SAME ENTRY FOUND!" +str(i+1))
				#merge_dict
				#f=0
				#break
				
			i+=1
		print("Repeated key found :" + str(f))
		
		print("New key found :" + str(n))
		if f==0 :
			print("Innner break!")

		print("length of resultant lexicon: "+ str(len(result)))
	return result
	


r = build_lexicon(dict_args)
print("Wrtiting to file....")
filename = 'hindi_lexicon.tff'
lines = []
for key in r:
	lines.append(str(key)+'\t'+ str(r[key]))

utils.save_file(lines, filename)

print('Lexicon generated.')


#print("Final "+str(len(r)))
	#print(list(splits))
#result = utils.get_subj_lexicon()

#for line in pos_c[:5]:
#	print(len(line.split('\t',1)))

#print(len(pos_c))



#print("Length of dictionary: %d " %(len(result)))


#pos_c = pos_c[54:]






































'''
lexic = {}
i=0
for line in pos_c:
	#print(len(line.split('\t')))
	try:
		pos, word = re.split(r'\t+',line)
	except:
		print("issue at line "+str(i+1))
		#pos, word = re.split(r'\t+', line)

	if word not in lexic:
		lexic[word]={}
		lexic[word][pos_dic[pos]]=['ambiguous']
	else:
		lexic[word][pos_dic[pos]]=['ambiguous']
	i+=1


for i in lexic:
	print (i +"\t"+str(lexic[i]))
'''

'''
neg_c = utils.load_file("")
neu_c = utils.load_file("")
amb_c = utils.load_file("")

doc_complete = [pos_c, neg_c, neu_c, amb_c]
'''