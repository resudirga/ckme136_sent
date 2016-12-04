# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 18:26:53 2016

@author: renita
"""
import os
import numpy as np
from sklearn.cluster import KMeans

#import pandas as pd
#import matplotlib.pyplot as plt
#import baseline_models as mm

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.cluster.kmeans import KMeansClusterer

from   sklearn.svm import LinearSVC
from   sklearn.ensemble import RandomForestClassifier

cdir = os.path.dirname(os.path.realpath(__file__))
stemmer=PorterStemmer()

def load_socal_dicts():
    adj_dict, adv_dict, noun_dict, verb_dict = dict(), dict(), dict(), dict()
    with open(cdir+'/SO-CAL/adj_dictionary1.11.txt') as f:
        for line in f:
            word, score = line.split()
            word = stemmer.stem(word)
            adj_dict[word] = score
        
    with open(cdir+'/SO-CAL/adv_dictionary1.11.txt') as f:
        for line in f:
            word, score = line.split()
            word = stemmer.stem(word)
            adv_dict[word] = score
        
    with open(cdir+'/SO-CAL/noun_dictionary1.11.txt') as f:
        for line in f:
            word, score = line.split()
            word = stemmer.stem(word)
            noun_dict[word] = score
               
    with open(cdir+'/SO-CAL/verb_dictionary1.11.txt') as f:
        for line in f:
            word, score = line.split()
            word = stemmer.stem(word)
            verb_dict[word] = score
            
    return adj_dict, adv_dict, noun_dict, verb_dict

class Converter(object):
    
    adj_dict, adv_dict, noun_dict, verb_dict = load_socal_dicts()
            
    def get_words_score(self, text):
        
        words_score = []        

        for word, tag in pos_tag(word_tokenize(text),tagset='universal'):
            word = stemmer.stem(word)
            score = 0.
            if   tag == 'VERB' and word in self.verb_dict.keys(): score = float(self.verb_dict[word])
            elif tag == 'ADJ'  and word in self.adj_dict.keys() : score = float(self.adj_dict[word])
            elif tag == 'ADV'  and word in self.adv_dict.keys() : score = float(self.adv_dict[word])    
            elif tag == 'NOUN' and word in self.noun_dict.keys(): score = float(self.noun_dict[word])
            else: 
                word = 'unk'
                score = 0.            
            words_score.append((word, score))
        return words_score
        
    def get_words_sum(self,text):

        tokens = self.get_words_score(text)        
        return sum([score for _,score in tokens])
        
        
    def make_ngram_array(self,text,ngram=2):
        
        tokens = self.get_words_score(text)
        tokens = [(word, score) for (word, score) in tokens if word != 'unk']
        
        if len(tokens) > 0:
            ngram_arr = np.zeros((len(tokens),ngram),dtype='float')    
            
            ngram_vec = np.zeros((ngram,),dtype='float')
            ngram_vec_prev = np.zeros((ngram,),dtype='float')
            
            for ix, (_, score) in enumerate(tokens):
                ngram_vec = np.append(ngram_vec_prev[1:], score)
                ngram_vec_prev = ngram_vec 
                ngram_arr[ix] = ngram_vec
        else: ngram_arr = []
            
        return ngram_arr
        
    def make_ngram_array_docs(self,docs,ngram=2):
        
        # Estimate the total number of tokens in the document
        n_tokens = 0        
        for text in iter(docs):
            n_tokens += len(word_tokenize(text))
        
        # Pre-allocate the np array
        ngram_arr = np.zeros((n_tokens,ngram),dtype='float')    
        
        ix_rows = 0
        for text in iter(docs):
            text_ngram =self.make_ngram_array(text,ngram)
            nrows = len(text_ngram)
            if nrows > 0:
                ngram_arr[ix_rows:ix_rows + nrows] = text_ngram
                ix_rows = ix_rows + nrows
        
        ngram_arr = ngram_arr[:ix_rows] if ix_rows > 0 else []
        
        return ngram_arr
        
    def get_ngram_avg(self,text,ngram=2):
        
        ngram_arr = self.make_ngram_array(text,ngram)
        if len(ngram_arr) > 0:
            ngram_avg = np.mean(ngram_arr,axis=0)
        else: ngram_avg = np.zeros((ngram,))
        return ngram_avg
        
    def makefeatures_sum(self,docs):        
        feature = [self.get_words_sum(text) for text in docs]
        feature = np.asarray(feature).reshape(-1,1)
        return feature
        
    def makefeatures_ngram_avg(self,docs,ngram=2):
        
        feature_mat = np.zeros((len(docs),ngram))
        for ix,text in enumerate(docs):
            feature_mat[ix] = self.get_ngram_avg(text,ngram)
        feature_mat[np.isnan(feature_mat)] = 0.        
        return feature_mat
        
    def makefeatures_clusters(self,train_docs,ngram=2,n_clusters=100):
        
        train_features = self.make_ngram_array_docs(train_docs,ngram)
        
        kmeans = KMeans(n_clusters=n_clusters,n_init=10, max_iter=300,n_jobs=-1)
        kmeans.fit_predict(train_features)
        
        train_clusters = np.array([ cluster for label,cluster in zip(kmeans.labels,kmeans.cluster_centers_) ] )
        
        return train_clusters
        
        
if __name__ == '__main__':
    
    # Writing the result into a text file
    txtfile = open('socal_featSelect.txt', 'w')
    
    # Load the dataset
    import pickle        
    reviews = pickle.load( open(os.path.join(cdir, 'reviews_munged.pkl'),'rb') )
    
    nfolds = 5
    ngram_opts = [0, 2, 5, 10]
    
    prediction_accuracy_clf1 = np.zeros([nfolds, len(ngram_opts)])
    prediction_accuracy_clf2 = np.zeros([nfolds, len(ngram_opts)])
    prediction_accuracy_clf3 = np.zeros([nfolds, len(ngram_opts)])
    
    T = Converter()
    
    for ix_fold in range(nfolds):
        train_fname = 'fold' + str(ix_fold) + '_train_SentenceId.pkl'
        valid_fname = 'fold' + str(ix_fold) + '_valid_SentenceId.pkl'
        train_sentids = pickle.load( open(os.path.join(cdir, train_fname),'rb') )
        valid_sentids = pickle.load( open(os.path.join(cdir, valid_fname),'rb') )
        
        fold_train_df = reviews[reviews['SentenceId'].isin(train_sentids)]
        fold_validation_df = reviews[reviews['SentenceId'].isin(valid_sentids)]
            
        train_docs = fold_train_df['Cleaned_Phrase'].tolist()
        train_labels = fold_train_df['Sentiment'].values
        validation_docs = fold_validation_df['Cleaned_Phrase'].tolist()
        validation_labels = fold_validation_df['Sentiment'].values
        
        for (ix_feature, ngram) in enumerate(ngram_opts):
            
            if ngram == 0:
                print('N-gram = %d: \n' %ngram)
                txtfile.write('Using sum of word scores as features: \n')
                train_feature_mat = T.makefeatures_sum(train_docs)
                validation_feature_mat = T.makefeatures_sum(validation_docs)
            else:
                print('N-gram = %d: \n' %ngram)
                txtfile.write('word-score ngram, n-gram = %d\n' %ngram)
                
                train_feature_mat = T.makefeatures_ngram_avg(train_docs,ngram=ngram)        
                validation_feature_mat = T.makefeatures_ngram_avg(validation_docs,ngram=ngram)
        
            clf1 = LinearSVC()
            clf1.fit(train_feature_mat,train_labels)
            prediction = clf1.predict(validation_feature_mat) 
            accuracy = sum((prediction - validation_labels) == 0) / len(validation_labels) * 100
            txtfile.write('Accuracy of Linear SVC classifier is ' + str(accuracy) + '%') #52.65
            
            prediction_accuracy_clf1[ix_fold, ix_feature] = accuracy
            
            del clf1
            
            clf2 = RandomForestClassifier(n_estimators=10,n_jobs=-1,min_samples_leaf=10)
            clf2.fit(train_feature_mat,train_labels)
            prediction = clf2.predict(validation_feature_mat) 
            accuracy = sum((prediction - validation_labels) == 0) / len(validation_labels) * 100
            txtfile.write('Accuracy of a Random Forest classifier (ntrees=10) is ' + str(accuracy) + '%')
            
            prediction_accuracy_clf2[ix_fold, ix_feature] = accuracy
             
            del clf2
            
            clf3 = RandomForestClassifier(n_estimators=50,n_jobs=-1,min_samples_leaf=10)
            clf3.fit(train_feature_mat,train_labels)
            prediction = clf3.predict(validation_feature_mat) 
            accuracy = sum((prediction - validation_labels) == 0) / len(validation_labels) * 100
            txtfile.write('Accuracy of a Random Forest classifier (ntrees=50) is ' + str(accuracy) + '%')                
            
            prediction_accuracy_clf3[ix_fold, ix_feature] = accuracy
             
            del clf3
    
            print('---------------------------------------\n\n')    
            
    txtfile.close()
    
    with open(os.path.join(cdir, 'socal_5folds_svc_validation_acc'),'wb') as f1, open(os.path.join(cdir, 'socal_5folds_forest10_validation_acc'),'wb') as f2, open(os.path.join(cdir, 'socal_5folds_forest50_validation_acc'),'wb') as f3:
        pickle.dump(prediction_accuracy_clf1, f1)
        pickle.dump(prediction_accuracy_clf2, f2)
        pickle.dump(prediction_accuracy_clf3, f3)