# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 21:07:36 2016
Generate csv files, columns = ['PhraseId', 'Sentiment'] to be submitted for Kaggle evaluation

@author: renita
"""

import ckme136_utl as u
import socal_transform as s
import bagofwords_models as bow
import word2vec_models_sentdivide as w2v
from gensim.models.word2vec import LineSentence

import os
import numpy as np
import pandas as pd
import pickle
import time

from   sklearn.svm import LinearSVC
from   sklearn.ensemble import RandomForestClassifier

cdir = os.path.dirname(os.path.realpath(__file__))

# Load the dataset
train = pickle.load( open(os.path.join(cdir, 'reviews_munged.pkl'),'rb') )

# Load the dataset and divide into train, test sets
test = pd.read_table(cdir + '/data/test.tsv',sep='\t'
                                      ,dtype={ 'PhraseId': str
                                            ,'SentenceId': str
                                            ,'Phrase': str} )
    
# append a column called 'Cleaned_Phrase' containing Phrase with removed stopwords, transformed into lowercase, etc                                           
test = u.munge(test)    

#-----------------------------------------
# Bag of words, n_words = 5000, Linear SVC
#-----------------------------------------
n_words = 5000
train_feature_matrix, tfidf_vect = bow.get_train_feature_matrix(train,tfidf=True,as_sparse=True, max_features=n_words)
train_labels = np.array(train['Sentiment'])

test_feature_matrix = bow.get_test_feature_matrix(test, tfidf_vect,as_sparse=True)

clf_bow_svc = LinearSVC()
start = time.time() 
clf_bow_svc.fit(train_feature_matrix, train_labels)
end = time.time()
elapsed = end - start

print('Time required to train the model: %.2f sec' %elapsed)

prediction = clf_bow_svc.predict(test_feature_matrix)

test['bow_prediction'] = prediction

test.to_csv(os.path.join(cdir, 'test_prediction_clf_bow_svc_f5000.csv'),columns=['PhraseId', 'bow_prediction'],header=['PhraseId','Sentiment'],index=False)

# Score = 0.60537
# Time required to train the model: 7.694647550582886 sec

#-----------------------------------------
# SO-CAL sum of word scores, Random Forest (ntrees=10)
#-----------------------------------------
train_docs = train['Cleaned_Phrase'].tolist()
train_labels = train['Sentiment'].values
test_docs = test['Cleaned_Phrase'].tolist()

T = s.Converter()
train_feature_vec = T.makefeatures_sum(train_docs)
test_feature_vec = T.makefeatures_sum(test_docs)

clf_socal_forest10 = RandomForestClassifier(n_estimators=10,n_jobs=-1,min_samples_leaf=10)
start = time.time() 
clf_socal_forest10.fit(train_feature_vec,train_labels)
end = time.time()
elapsed = end - start

print('Time required to train the model: %.2f sec' %elapsed)

prediction = clf_socal_forest10.predict(test_feature_vec) 

test['socal_prediction'] = prediction
test['socal_forest10_proba'] = clf_socal_forest10.predict_proba(test_feature_vec).tolist()

test.to_csv(os.path.join(cdir, 'test_prediction_clf_socalsum_forest10.csv'),columns=['PhraseId', 'socal_prediction'],header=['PhraseId','Sentiment'],index=False)

# Score = 0.56364
# Time required to train the model: 0.35 sec

#-----------------------------------------
# Word2Vec vectors, trained on SAR14 dataset (IMDB data with 233k reviews), Random Forest (ntrees=100)
#-----------------------------------------

### Run the following commented block of code to train the word2vec model
#sentences = LineSentence('cleaned_sar14.csv')
#
## Word2Vec parameters
#num_features=300
#min_word_count = 10   # Minimum word count
#num_workers = 2       # Number of threads to run in parallel
#win_len = 10          # Context window size; this is the recommended win_size for CBOW
#downsampling = 1e-3   # Downsample setting for frequent words 
#model = w2v.Word2Vec(sentences, workers=num_workers, \
#                    size=num_features, min_count = min_word_count, \
#                    window = win_len, sample = downsampling)
#with open(os.path.join(cdir, 'word2vec_sar14_Data.pkl'),'wb') as f1:
#    pickle.dump(model, f1)

# Load the pre-trained model
model = pickle.load( open(os.path.join(cdir, 'word2vec_sar14_Data.pkl'),'rb') )

# Get the training and test feature matrices and labels
num_features = 300
train = pickle.load( open(os.path.join(cdir, 'reviews_munged.pkl'),'rb') )
train_docs_df = train[['Cleaned_Phrase', 'Sentiment' ]].drop_duplicates()      # since we remove stop words, some of the phrases and sentiment become identical-keep only 1 of them 
train_docs = train_docs_df['Cleaned_Phrase'].apply(lambda sent: sent.split()).tolist()
train_labels = train_docs_df['Sentiment'].values

test_docs = test['Cleaned_Phrase'].apply(lambda sent: sent.split()).tolist()
                        
train_feature_matrix = w2v.makefeatures_avg(train_docs, model, num_features) 
test_feature_matrix = w2v.makefeatures_avg(test_docs, model, num_features)

clf_w2v_forest100 = RandomForestClassifier(n_estimators=100,n_jobs=-1,min_samples_leaf=10)
start = time.time() 
clf_w2v_forest100.fit(train_feature_matrix,train_labels)
end = time.time()
elapsed = end - start

print('Time required to train the model: %.2f sec' %elapsed)

prediction = clf_w2v_forest100.predict(test_feature_matrix)

test['w2v_n300_prediction'] = prediction
test['w2v_n300_proba'] = clf_w2v_forest100.predict_proba(test_feature_matrix).tolist()

test.to_csv(os.path.join(cdir, 'test_prediction_clf_w2v_n300_forest100.csv'),columns=['PhraseId', 'w2v_n300_prediction'],header=['PhraseId','Sentiment'],index=False)

# Score: 0.59162 

# Save the model so we can do more analysis
with open(os.path.join(cdir, 'final_clf_tfidf.pkl'),'wb') as f1, open(os.path.join(cdir, 'final_clf_socal.pkl'),'wb') as f2, open(os.path.join(cdir, 'final_clf_w2v.pkl'),'wb') as f3:
    pickle.dump(clf_bow_svc, f1)
    pickle.dump(clf_socal_forest10, f2)
    pickle.dump(clf_w2v_forest100, f3)


#-----------------------------------------
# Ensemble classifier
#-----------------------------------------
def getcls_maxproba(proba_list1, proba_list2):
    proba_cls1 = max(proba_list1)
    cls1 = proba_list1.index(proba_cls1)
    proba_cls2 = max(proba_list2)
    cls2 = proba_list2.index(proba_cls2)
    
    cls = cls1 if proba_cls1 > proba_cls2 else cls2
    return cls
    
test['ens1_prediction'] = test[['socal_forest10_proba','w2v_n100_proba']].apply(lambda x: getcls_maxproba(x[0], x[1]),axis=1)

