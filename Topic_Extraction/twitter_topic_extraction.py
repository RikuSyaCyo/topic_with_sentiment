# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 

@author: zhusd

"""

# Example run:
# python twitter_topic_extraction.py TXcon.csv TX.csv ERROR.txt 
# TXcon.csv stands for the input csv file(intput structure refering to main function)
# TX.csv refers to the output file
# ERROR.txt refers to the error information
# note: CMUTweetTagger.py is given, please put these files together

import re
import os
import sys
import csv
import nltk
import string
import codecs
import fastcluster
import numpy as np
import pandas as pd
import CMUTweetTagger
from sklearn import metrics
import matplotlib.pylab as plt
from collections import Counter
from sklearn import preprocessing
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

debug = 0

def file_input(file_timeordered_tweets):
	window_corpus = []
	tid_to_raw_tweet = {}
	tids_window_corpus=[]

	for line in file_timeordered_tweets:

		[tweet_id, text,number_users,number_tags] = (line.strip('\r\n').split(','))
		tweet_id=eval(tweet_id)
		if spam_tweet(text):
			continue
		features = process_json_tweet(text, fout, debug)
		tweet_bag = ""
		for feature in features:
			tweet_bag+=feature +','

		if int(number_users) < 5 and int(number_tags) < 3 and len(features) > 3 and len(tweet_bag.split(",")) > 4 and not str(features).upper() == str(features):
			tweet_bag = tweet_bag[:-1]
			window_corpus.append(tweet_bag)
			tids_window_corpus.append(tweet_id)
			tid_to_raw_tweet[tweet_id] = text

	return  tweet_id, text,number_users,number_tags,window_corpus,tids_window_corpus,tid_to_raw_tweet,features


def load_stopwords():
	stop_words = nltk.corpus.stopwords.words('english')
	stop_words.extend(['this','that','the','might','have','been','from',
				'but','they','will','has','having','had','how','went'
				'were','why','and','still','his','her','was','its','per','cent',
				'a','able','about','across','after','all','almost','also','am','among',
				'an','and','any','are','as','at','be','because','been','but','by','can',
				'cannot','could','dear','did','do','does','either','else','ever','every',
				'for','from','get','got','had','has','have','he','her','hers','him','his',
				'how','however','i','if','in','into','is','it','its','just','least','let',
				'like','likely','may','me','might','most','must','my','neither','nor',
				'not','of','off','often','on','only','or','other','our','own','rather','said',
				'say','says','she','should','since','so','some','than','that','the','their',
				'them','then','there','these','they','this','tis','to','too','twas','us','ca',
				'wants','was','we','were','what','when','where','which','while','who',
				'whom','why','will','with','would','yet','you','your','ve','re','rt', 'retweet', '#fuckem', '#fuck',
				'fuck', 'ya', 'yall', 'yay', 'youre', 'youve', 'ass','factbox', 'com', '&lt', 'th',
				'retweeting', 'dick', 'fuckin', 'shit', 'via', 'fucking', 'im','shocker', 'wtf', 'hey', 'ooh', 'rt&amp', '&amp',
				'#retweet', 'retweet', 'goooooooooo', 'hellooo', 'gooo', 'fucks', 'fucka', 'bitch', 'wey', 'sooo', 'helloooooo', 'lol', 'smfh'])
	stop_words = set(stop_words)
	return stop_words



def normalize_text(text):
	try:
		text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', text)
		text = re.sub('@[^\s]+','', text)
		text = re.sub('#([^\s]+)', '', text)
		text = re.sub('[:;>?<=*+()/,\-#!$%\{˜|\}\[^_\\@\]1234567890’‘]',' ', text)
		text = re.sub('[\d]','', text)
		text = text.replace(".", '')
		text = text.replace("'", ' ')
		text = text.replace("\"", ' ')
		text = text.replace("\x9d",' ').replace("\x8c",' ')
		text = text.replace("\xa0",' ')
		text = text.replace("\x9d\x92", ' ').replace("\x9a\xaa\xf0\x9f\x94\xb5", ' ').replace("\xf0\x9f\x91\x8d\x87\xba\xf0\x9f\x87\xb8", ' ').replace("\x9f",' ').replace("\x91\x8d",' ')
		text = text.replace("\xf0\x9f\x87\xba\xf0\x9f\x87\xb8",' ').replace("\xf0",' ').replace('\xf0x9f','').replace("\x9f\x91\x8d",' ').replace("\x87\xba\x87\xb8",' ')   
		text = text.replace("\xe2\x80\x94",' ').replace("\x9d\xa4",' ').replace("\x96\x91",' ').replace("\xe1\x91\xac\xc9\x8c\xce\x90\xc8\xbb\xef\xbb\x89\xd4\xbc\xef\xbb\x89\xc5\xa0\xc5\xa0\xc2\xb8",' ')
		text = text.replace("\xe2\x80\x99s", " ").replace("\xe2\x80\x98", ' ').replace("\xe2\x80\x99", ' ').replace("\xe2\x80\x9c", " ").replace("\xe2\x80\x9d", " ")
		text = text.replace("\xe2\x82\xac", " ").replace("\xc2\xa3", " ").replace("\xc2\xa0", " ").replace("\xc2\xab", " ").replace("\xf0\x9f\x94\xb4", " ").replace("\xf0\x9f\x87\xba\xf0\x9f\x87\xb8\xf0\x9f", "")
	except:pass
	return text



def nltk_tokenize(text):
	tokens = []
	pos_tokens = []
	entities = []
	features = []
	stop_words = load_stopwords()
	try:
		tokens = text.split()
		for word in tokens:
			if word.lower() not in stop_words and len(word) > 1:        
				features.append(word)   
	except: pass
	return [tokens, pos_tokens, entities, features] 




def process_json_tweet(text,fout,debug):
	features = []
	if (len(text.strip()) == 0):
		return []
	text = normalize_text(text)
	try:
		[tokens,pos_tokens,entities,features] = nltk_tokenize(text)

	except: print("nltk tokenize + pos pb!!!")
	if debug:	
		try:
			fout.write("\n--------------------clean text--------------------\n")
			fout.write(text.decode('utf-8'))
			fout.write("\n--------------------tokens--------------------\n")
			fout.write(str(tokens))
			fout.write("\n--------------------pos tokens--------------------\n")
			fout.write(str(pos_tokens))
			fout.write("\n--------------------entities--------------------\n")
			for ent in entities:
				fout.write("\n" + str(ent).decode('utf-8'))
			fout.write("\n--------------------features--------------------\n")
			fout.write(str(features))
			fout.write("\n\n")
		except:
			print ("couldn't print text")
			pass
	return features


	
'''Prepare features, where doc has terms separated by comma'''

def custom_tokenize_text(text):
	REGEX = re.compile(r",\s*")
	tokens = []
	for tok in REGEX.split(text):
		if "@" not in tok:
			tokens.append(tok.strip().lower())
	# print(tokens)
	return tokens



def spam_tweet(text):
	spam_words = ['work', 'job', 'mph', 'join', 'ca', 'ny', 'airport']
	if 'new job' in text:
		return True
	if 'jobsjdhuntr' in text:
		return True
		
	if 'latest opening' in text:
		return True

	if 'want to work' in text:

		return True

	if 'follow me please' in text:
		return True
	
	if 'please follow me' in text:
		return True	

	for word in spam_words:
		if word in text:
			return True	
	return False



def tweet_cluster(window_corpus):

	vectorizer = CountVectorizer(tokenizer=custom_tokenize_text, binary=True, min_df=max(int(len(window_corpus)*0.0025), 10), ngram_range=(2,3))
	X = vectorizer.fit_transform(window_corpus)
	# print(features)
	map_index_after_cleaning = {}
	Xclean = np.zeros((1, X.shape[1]))
	for i in range(0, X.shape[0]):
		if X[i].sum() > 4:
			Xclean = np.vstack([Xclean, X[i].toarray()])
			map_index_after_cleaning[Xclean.shape[0] - 2] = i
	Xclean = Xclean[1:,]
	X = Xclean
	return X,map_index_after_cleaning,vectorizer

def boost_df(X):
	boost_entity = {}
	boosted_wdfVoc = {} 
	Xdense = np.matrix(X).astype('float')
	X_scaled = preprocessing.scale(Xdense)
	X_normalized = preprocessing.normalize(X_scaled, norm='l2')
	vocX = vectorizer.get_feature_names()

	pos_tokens = CMUTweetTagger.runtagger_parse([term.upper().encode() for term in vocX])

	for l in pos_tokens:
		term =''
		for gr in range(0, len(l)):
			term += l[gr][0].lower() + " "
		if "^" in str(l):
			boost_entity[term.strip()] = 2.5
		else:           
			boost_entity[term.strip()] = 1.0

	dfX = X.sum(axis=0)
	keys = vocX
	vals = dfX
	for k,v in zip(keys, vals):
		dfVoc[k] = v
	for k in dfVoc:
		try:
			boosted_wdfVoc[k] = dfVoc[k] * boost_entity[k]
		except:
			boosted_wdfVoc[k] = dfVoc[k]

	return X_normalized,boosted_wdfVoc

def sort_tweet_clusters(X_normalized,X):

	#fastcluster, average, cosine					
	distMatrix = pairwise_distances(X_normalized, metric='cosine')
	L = fastcluster.linkage(distMatrix, method='average')
	P=sch.dendrogram(L)
	plt.savefig('voc_dendrogram.png')
	# hclust cut threshold
	dt = 0.8
	indL = sch.fcluster(L, dt*distMatrix.max(), 'distance')
	freqTwCl = Counter(indL)
	npindL = np.array(indL)


	#frequence of bi-gram/tri-gram 
	freq_th = max(10, int(X.shape[0]*0.0025))
	cluster_score = {}
	for clfreq in freqTwCl.most_common(50):
		cl = clfreq[0]
		freq = clfreq[1]
		cluster_score[cl] = 0
		if freq >= freq_th:
			clidx = (npindL == cl).nonzero()[0].tolist()
			cluster_centroid = X[clidx].sum(axis=0)
			try:
				cluster_tweet = vectorizer.inverse_transform(cluster_centroid)
				for term in np.nditer(cluster_tweet):
					try:
						cluster_score[cl] = max(cluster_score[cl], boosted_wdfVoc[str(term).strip()])
					except: pass            
			except: pass
			cluster_score[cl] /= freq
		else: break
	sorted_clusters = sorted( ((v,k) for k,v in cluster_score.items()), reverse=True)
	
	return sorted_clusters,npindL,cluster_score

def headline_cluster(headline_corpus):
	# cluster headlines to avoid topic repetition
	headline_vectorizer = CountVectorizer(tokenizer=custom_tokenize_text, binary=True, min_df=1, ngram_range=(1,1))
	H = headline_vectorizer.fit_transform(headline_corpus)
	vocH = headline_vectorizer.get_feature_names()
	Hdense = np.matrix(H.todense()).astype('float')
	distH = pairwise_distances(Hdense, metric='cosine')
	HL = fastcluster.linkage(distH, method='average')
	p_h=sch.dendrogram(HL)
	plt.savefig('headline_dendrogram.png')
	dtH = 1.0
	indHL = sch.fcluster(HL, dtH*distH.max(), 'distance')
	freqHCl = Counter(indHL)

	npindHL = np.array(indHL)
	hcluster_score = {}
	for hclfreq in freqHCl.most_common(ntopics):
		hcl = hclfreq[0]
		hfreq = hclfreq[1]
		hcluster_score[hcl] = 0
		hclidx = (npindHL == hcl).nonzero()[0].tolist()
		for i in hclidx:
			hcluster_score[hcl] = max(hcluster_score[hcl], cluster_score[headline_to_cluster[headline_corpus[i]]])
	sorted_hclusters = sorted( ((v,k) for k,v in hcluster_score.items()), reverse=True)

	return sorted_hclusters,npindHL

def sort_headline_clusters(sorted_hclusters):
	headline_final = []
	keywords_each_headline = []
	tid_each_headline = []
	for hscore, hcl in sorted_hclusters[:10]:
		hclidx = (npindHL == hcl).nonzero()[0].tolist()
		clean_headline = ''
		raw_headline = ''
		keywords = ''
		tids_set = set()
		tids_list = []
		selected_raw_tweets_set = set()
		tids_cluster = []
	
		for i in hclidx:
			clean_headline += headline_corpus[i].replace(",", " ") 
			keywords += orig_headline_corpus[i].lower() + ","
			tid = headline_to_tid[headline_corpus[i]]
			tids_set.add(tid)
			raw_tweet = tid_to_raw_tweet[tid].replace("\n", ' ').replace("\t", ' ')
			raw_tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', raw_tweet)
			selected_raw_tweets_set.add(raw_tweet.strip())
			tids_list.append(tid)

			for id in cluster_to_tids[headline_to_cluster[headline_corpus[i]]]:
				tids_cluster.append(id)

		raw_headline = tid_to_raw_tweet[headline_to_tid[headline_corpus[hclidx[0]]]]
		raw_headline = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', raw_headline)
		raw_headline = raw_headline.replace("\n", ' ').replace("\t", ' ')
		keywords_list = str(sorted(list(set(keywords[:-1].split(",")))))[1:-1].replace('u\'','').replace('\'','')
		headline_final.append(raw_headline);


		for tid in tids_cluster:
			raw_tweet = tid_to_raw_tweet[tid].replace("\n", ' ').replace("\t", ' ')
			raw_tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', raw_tweet)
			if raw_tweet.strip() not in selected_raw_tweets_set:
				tids_list.append(tid)
				selected_raw_tweets_set.add(raw_tweet.strip())
		
		keywords_each_headline.append(clean_headline)
		tid_each_headline.append(tids_list)

	return headline_final,keywords_each_headline,tid_each_headline

def selec_top_cluster(ntopics,sorted_clusters,window_corpus):
	headline_corpus = []
	orig_headline_corpus = []
	headline_to_cluster = {}
	headline_to_tid = {}
	cluster_to_tids = {}
	for score,cl in sorted_clusters[:ntopics]:
		#twitter index of each cluster
		clidx = (npindL == cl).nonzero()[0].tolist()
		first_idx = map_index_after_cleaning[clidx[0]]
		keywords = window_corpus[first_idx]
		orig_headline_corpus.append(keywords)
		headline = ''
		for k in keywords.split(","):
			if not '@' in k and not '#' in k:
				headline += k + ","
		headline_corpus.append(headline[:-1])
		headline_to_cluster[headline[:-1]] = cl
		headline_to_tid[headline[:-1]] = tids_window_corpus[first_idx]
		tids = []
		for i in clidx:
			idx = map_index_after_cleaning[i]
			tids.append(tids_window_corpus[idx])
		cluster_to_tids[cl] = tids 

	return headline_corpus,orig_headline_corpus,headline_to_cluster,headline_to_tid,cluster_to_tids



'''start main'''
if __name__=="__main__":
	file_timeordered_tweets=codecs.open(sys.argv[1], 'r', 'utf-8')
	fout = codecs.open(sys.argv[3], 'w', 'utf-8')

	tweet_id, text,number_users,number_tags,window_corpus,tids_window_corpus,tid_to_raw_tweet,features = file_input(file_timeordered_tweets)

	dfVoc = {}
	wdfVoc = {}
	boosted_wdfVoc = {} 

	X,map_index_after_cleaning,vectorizer= tweet_cluster(window_corpus)
	X_normalized,boosted_wdfVoc = boost_df(X)

	sorted_clusters,npindL,cluster_score = sort_tweet_clusters(X_normalized,X)


	ntopics = 20

	headline_corpus,orig_headline_corpus,headline_to_cluster,headline_to_tid,cluster_to_tids=selec_top_cluster(ntopics,sorted_clusters,window_corpus)

	sorted_hclusters,npindHL = headline_cluster(headline_corpus)

	headline_final,keywords_each_headline,tid_each_headline = sort_headline_clusters(sorted_hclusters)


	with open(sys.argv[2],"w",newline="") as datacsv:
		csvwriter = csv.writer(datacsv,dialect = ("excel"))
		for i in range(len(headline_final)):
			csvwriter.writerow(["custom_name",headline_final[i],keywords_each_headline[i],str(tid_each_headline[i])[1:-1]])
				
	# print(headline_final)
	file_timeordered_tweets.close()
	fout.close()
