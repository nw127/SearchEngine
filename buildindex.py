import re, os, time
from html.parser import HTMLParser
from collections import defaultdict, OrderedDict
import json, math
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

docCount = 0
uniqueWords = 0

class MyHTMLParser(HTMLParser):
	def __init__(self):
		HTMLParser.__init__(self)
		self.t = "title"
		self.alltags = []
		self.htmlDict = defaultdict(str)
		
	def handle_starttag(self, tag, attrs):
		self.alltags.append(tag)
		self.t = tag
	
	def handle_endtag(self, tag):
		if len(self.alltags) > 0:
			del self.alltags[-1]
			if len(self.alltags) > 0:
				self.t = self.alltags[-1]
		else:
			self.t = ""
	
	def handle_data(self, data):
		self.htmlDict[self.t] = self.htmlDict[self.t] + data

	def parseHTML(self, f):
		html = f.read()
		self.feed(html)
		#print(self.htmlDict)
		
		
		
class InvertedIndex:
	def __init__(self):
		self.ii = {}
		self.tf = defaultdict(lambda: defaultdict(float))
		self.df = defaultdict(int)
		self.uv = defaultdict(float)
		self.idf = defaultdict(float)
		self.ai = defaultdict(lambda: defaultdict(list))
		self.tfidf = defaultdict(lambda: defaultdict(float))
		

def parseFiles(d):
	"""go through all files and create dict. key is docID and value is parser results"""
	global docCount
	result = {}
	subdirs = [x[0] for x in os.walk(d)]
	for subdir in subdirs:
		if subdir != d and os.path.isdir(subdir):
			print(subdir)
			files = next(os.walk(subdir))[2]
			if (len(files) > 0):
				for f in files:
					if not f.startswith("."):
						#print(f)
						docID = os.path.basename(os.path.normpath(subdir)) + "/" + f
						openFile = open(subdir+ "/" + f, "r")
						parser = MyHTMLParser()
						parser.parseHTML(openFile)
						cleanText(parser)
						result[docID] = parser
						openFile.close()
						docCount += 1
	return result

def cleanText(p):
	"""take each doc's parsed text and clean it of white space and stopwords
	then stem each word."""
	stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought","our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
	ps = PorterStemmer()
	for k,v in p.htmlDict.items():
		pattern = re.compile('[\W_]+')
		p.htmlDict[k] = v.lower()
		p.htmlDict[k] = pattern.sub(' ', p.htmlDict[k])
		re.sub(r'[\W_]+','', p.htmlDict[k])
		p.htmlDict[k] = p.htmlDict[k].split()
		p.htmlDict[k] = [w for w in p.htmlDict[k] if w not in stopwords]
		p.htmlDict[k] = [ps.stem(w) for w in p.htmlDict[k]]
			
def combineIndices(dicts):
	"""combine all indices into a dict. key is docID and value is index for the file"""
	result = {}
	for k,v in dicts.items():
		result[k] = fileIndex(v, k)
	return result

def fileIndex(p, did):
	"""build index. key is word. value is list of position of word in doc"""
	d = defaultdict(list)
	for k,v in p.htmlDict.items():
		if k == "body":
			for i, w in enumerate(v):
				d[w].append(i)
		else:
			for i, w in enumerate(v):
				d[w+"!"].append(i)
	return d

def buildIndex(allIndices, obj):
	"""input is dict with all the indices where key is docID and value is index for the file.
	output is the inverted index with key as word and value as dict with key as filename
	and value as a list of the word's positions in the file"""
	global uniqueWords
	result = defaultdict(lambda : defaultdict(list))
	for f in allIndices.keys():
		for word in allIndices[f].keys():
			obj.tf[f][word] = len(allIndices[f][word])
			obj.df[word] += 1
			if word in result.keys():
				if f in result[word].keys():
					result[word][f].extend(allIndices[f][word][:])
				else:
					result[word][f] = allIndices[f][word]
			else:
				if word+"!" not in result.keys():
					uniqueWords += 1
				result[word] = {f: allIndices[f][word]}
	return result

def storeIndex(ii):
	with open("InvertedIndex.txt", "w") as f:
		json.dump(ii, f)
		
def calculateTF(obj):
	"""calculate term frequency for the index"""
	for f in obj.ai.keys():
		if f in obj.tf.keys():
			num = len(obj.tf[f].keys())
			for word in obj.tf[f].keys():
				obj.tf[f][word] = obj.tf[f][word]/num

def calculateIDF(obj):
	"""calculated inverse document frequency"""
	c = len(obj.ai.keys())
	for word in obj.ii.keys():
		if word in obj.df.keys():
			obj.idf[word] = math.log(c/obj.df[word]) if obj.df[word] != 0 else 0
		else:
			obj.idf[word] = 0

def calculateTFIDF(obj):
	"""calculates tf-idf for each word"""
	for f in obj.ai.keys():
		for word in obj.ai[f].keys():
			obj.tfidf[f][word] = obj.tf[f][word] * obj.idf[word]

def singleQuery(w, ii):
	"""query for single word. cut out white space and stem word first.
	get all the files that contain the word."""
	pattern = re.compile('[\W_]+')
	w = pattern.sub(' ',w)
	ps = PorterStemmer()
	w = ps.stem(w.lower())
	docIDs = list()
	if w in ii.keys():
		docIDs = [f for f in ii[w].keys()]
	if w+"!" in ii.keys():
		docIDs += [f for f in ii[w+"!"].keys()]
	return docIDs
	
def query(string, ii):
	"""returns set of files that contain all the words in the query"""
	docIDs = []
	for word in string.split():
		docIDs.append(singleQuery(word, ii))
	return set(docIDs[0]).intersection(*docIDs)

def queryTFIDF(string, docIDs, obj):
	"""calculate tf-idf for the query"""
	tfidf = defaultdict(float)
	for word in set(string.split()):
		tfidf[word] = string.split().count(word)/len(string.split()) * obj.idf[word]
	return tfidf

def similarity(string, tfidf, docIDs, obj):
	"""calculates a score for each doc relative to the query"""
	"""multiply score by 3 if word is in title/header/bold"""
	result = defaultdict(float)
	for f in docIDs:
		dp = 0
		q = 0
		d = 0
		for word in string.split():
			x = 0
			if word+"!" in obj.tfidf[f].keys():
				x =  obj.tfidf[f][word+"!"] * 3
			else:
				x =  obj.tfidf[f][word]
			dp += (tfidf[word] * x)
			q += (tfidf[word] * tfidf[word])
			d += (x * x)
		q = math.sqrt(q)
		d = math.sqrt(d)
		sim = dp/q*d
		result[f] = sim
	return result

def cleanQuery(string):
	"""remove white space, punctuation, stop words. then stems word"""
	stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought","our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
	ps = PorterStemmer()
	pattern = re.compile('[\W_]+')
	string = string.lower()
	string = pattern.sub(' ', string)
	re.sub(r'[\W_]+','', string)
	result = []
	for word in string.split():
		if word not in stopwords:
			result.append(ps.stem(word))
	return " ".join(result)

def getURLs(docIDs, bk):
	"""gets all urls using docIDs and bookkeeping.json file"""
	result = []
	for d in docIDs:
		result.append(bk[d])
	return result


if __name__ == "__main__":
	AllData = InvertedIndex()
	
	#build index and store in text file
	cwd = os.getcwd()
	dicts = parseFiles(cwd)
	allIndices = combineIndices(dicts)
	temp = buildIndex(allIndices, AllData)
	invertedIndex = OrderedDict(sorted(temp.items()))
	storeIndex(invertedIndex)
	print("Inverted index stored in text file.")
	print("Doc Count: " + str(docCount))
	print("Unique words: " + str(uniqueWords))
	
	#save term frequency, doc frequency, inverted doc freq
	#and if-idf after calculating
	AllData.ii = invertedIndex
	tf = OrderedDict(sorted(AllData.tf.items()))
	with open("tf.txt", "w") as f:
		json.dump(tf, f)
	
	df = OrderedDict(sorted(AllData.df.items()))
	with open("df.txt", "w") as f:
		json.dump(df, f)
	
	ai = OrderedDict(sorted(AllData.ai.items()))
	with open("indices.txt", "w") as f:
		json.dump(ai, f)
	
	calculateTF(AllData)
	tf = OrderedDict(sorted(AllData.tf.items()))
	with open("tf2.txt", "w") as f:
		json.dump(tf, f)
	
	calculateIDF(AllData)
	idf = OrderedDict(sorted(AllData.idf.items()))
	with open("idf.txt", "w") as f:
		json.dump(idf, f)
	
	calculateTFIDF(AllData)
	tfidf = idf = OrderedDict(sorted(AllData.tfidf.items()))
	with open("tfidf.txt", "w") as f:
		json.dump(tfidf, f)

	#load index
	bk = {}
	with open("bookkeeping.json") as j:
		bk = json.load(j)
	with open("InvertedIndex.txt") as f:
		AllData.ii = json.load(f)
	with open("idf.txt") as f:
		AllData.idf = json.load(f)
	with open("tfidf.txt") as f:
		AllData.tfidf = json.load(f)
	
	#run search engine
	print("Index Loaded")
	while True:
		s = input("Enter query: ")
		s = cleanQuery(s)
		dids = query(s, AllData.ii)
		if len(dids) > 0:
			tfidf = queryTFIDF(s, dids, AllData)
			r = similarity(s, tfidf, dids, AllData)
			asdf = OrderedDict(sorted(r.items(), key=lambda x: x[1], reverse=True))
			d = list(asdf.keys())
			pairs = list(asdf.items())
			urls = getURLs(d, bk)
			for i in range(len(d)):
				print("{0:.10f}".format(round(pairs[i][1], 10)) + "\turl: " + json.dumps(urls[i]))
		else:
			print("No files contained the query")

	