# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:07:02 2016

@author: renita
"""
import os
import logging
from gensim.models import Word2Vec
import numpy as np
import ckme136_utl as u
from   gensim.models.word2vec import LineSentence

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier  

cdir = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)
        
#def load_moviereviews_data():
#    """ 
#    Return all movie reviews data from nltk corpus as a LineSentence instance.
#    Data from: from nltk.corpus import movie_reviews    
#    """    
#    sentences = LineSentence('movie_reviews.txt')
#    return sentences


def get_avg_wordvecs(sentence, model, num_features):
    """
    Given a word2vec model and a sentence (a list of words), 
    return the average of the word vectors as an np-array
    """
    vocabs = set(model.index2word)
    featureAvg = np.zeros((num_features,), dtype=float)
    num_words = 0.
    
    for word in sentence:
        word = word.lower()
        if word in vocabs:
            num_words += 1.
            featureAvg = np.add(featureAvg,model[word])
    
    featureAvg = np.divide(featureAvg, num_words) if num_words > 0. else featureAvg      
    return featureAvg

def makefeatures_avg(doc, model, num_features):
    """
    Given a list of sentences (a document), return the average word2vec of each sentence in the list
    """    
    feature_mat = np.zeros((len(doc),num_features), dtype=float)
    for ix,sent in enumerate(doc):
        feature_mat[ix] = get_avg_wordvecs(sent, model, num_features)
    return feature_mat

if __name__ == '__main__':
   
   # Writing the result into a text file
    txtfile = open('word2vec_sar14_folds_featsel.txt', 'w')
    
    # Fixed word2vec parameters
    min_word_count = 10   # Minimum word count
    num_workers = 3       # Number of threads to run in parallel
    win_len = 10          # Context window size; this is the recommended win_size for CBOW
    downsampling = 1e-3   # Downsample setting for frequent words 
    
    # Size of the word2vec vectors to be evaluated
    num_features_opts = [100, 200, 300, 500, 1000]
    model_names = ['w2v_sar14_nfeat100', 'w2v_sar14_nfeat200','w2v_sar14_nfeat300','w2v_sar14_nfeat500','w2v_sar14_nfeat1000']
    
    nfolds = 5

    prediction_accuracy_clf2 = np.zeros([nfolds, len(num_features_opts)])
    prediction_accuracy_clf3 = np.zeros([nfolds, len(num_features_opts)])
    
    # Load the dataset
    import pickle        
    reviews = pickle.load( open(os.path.join(cdir, 'reviews_munged.pkl'),'rb') )
   
    # Pre-train the model
    sentences = LineSentence('cleaned_sar14.csv')
    model_sar14_100 = Word2Vec(sentences, workers=num_workers, \
                                size=100, min_count = min_word_count, \
                                window = win_len, sample = downsampling, seed=100)
    model_sar14_200 = Word2Vec(sentences, workers=num_workers, \
                                size=200, min_count = min_word_count, \
                                window = win_len, sample = downsampling, seed=100)
    model_sar14_300 = pickle.load( open(os.path.join(cdir, 'word2vec_sar14_Data.pkl'),'rb') )
    model_sar14_500 = Word2Vec(sentences, workers=num_workers, \
                                size=500, min_count = min_word_count, \
                                window = win_len, sample = downsampling, seed=100)
    model_sar14_1000 = Word2Vec(sentences, workers=num_workers, \
                                size=1000, min_count = min_word_count, \
                                window = win_len, sample = downsampling, seed=100)
            
    print('Completed training a Word2Vec models...') 
    
    for ix_fold in range(nfolds):
        train_fname = 'fold' + str(ix_fold) + '_train_SentenceId.pkl'
        valid_fname = 'fold' + str(ix_fold) + '_valid_SentenceId.pkl'
        train_sentids = pickle.load( open(os.path.join(cdir, train_fname),'rb') )
        valid_sentids = pickle.load( open(os.path.join(cdir, valid_fname),'rb') )
        
        fold_train_df = reviews[reviews['SentenceId'].isin(train_sentids)]
        fold_train_df = fold_train_df[['Cleaned_Phrase', 'Sentiment' ]].drop_duplicates()
        fold_validation_df = reviews[reviews['SentenceId'].isin(valid_sentids)]
        fold_validation_df = fold_validation_df[['Cleaned_Phrase', 'Sentiment' ]].drop_duplicates()
            
        train_docs = fold_train_df['Cleaned_Phrase'].apply(lambda sent: sent.split()).tolist()
        train_labels = fold_train_df['Sentiment'].values
        validation_docs = fold_validation_df['Cleaned_Phrase'].apply(lambda sent: sent.split()).tolist()
        validation_labels = fold_validation_df['Sentiment'].values
        
        for (ix_feature, num_features) in enumerate(num_features_opts):
            print('Num_features = %d: \n' %num_features)
            
            if num_features == 100:  model =  model_sar14_100
            elif num_features == 200: model =  model_sar14_200
            elif num_features == 300: model =  model_sar14_300
            elif num_features == 500: model =  model_sar14_500
            else : model =  model_sar14_1000                    
    
            # Make the features
            train_feature_matrix = makefeatures_avg(train_docs, model, num_features) 
            validation_feature_matrix = makefeatures_avg(validation_docs, model, num_features)
        
            del model
            
            # Choosing the classifier
            clf2 = LinearSVC()
            clf2.fit(train_feature_matrix, train_labels)
            prediction = clf2.predict(validation_feature_matrix)
            accuracy = u.get_classification_accuracy(prediction, validation_labels)
            
            del clf2
            
            prediction_accuracy_clf2[ix_fold, ix_feature] = accuracy
            txtfile.write('Accuracy of Linear SVC is %s\n' %accuracy) 
            print('Accuracy of Linear SVC is %s\n' %accuracy)
            
            # Choosing the classifier
            clf3 = RandomForestClassifier(n_estimators=100,n_jobs=-1,min_samples_leaf=10)
            clf3.fit(train_feature_matrix, train_labels)
            prediction = clf3.predict(validation_feature_matrix)
            accuracy = u.get_classification_accuracy(prediction, validation_labels)
            
            del clf3
            
            prediction_accuracy_clf3[ix_fold, ix_feature] = accuracy
            txtfile.write('Accuracy of Random Forest (ntrees=100) is %s\n' %accuracy) 
            print('Accuracy of Random Forest (ntrees=100) is %s\n' %accuracy)
    
            print('---------------------------------------\n\n')    
            
    txtfile.close()
    
    with open(os.path.join(cdir, 'w2v_sar14_5folds_svc_validation_acc'),'wb') as f2, open(os.path.join(cdir, 'w2v_sar14_5folds_forest_validation_acc'),'wb') as f3:
            pickle.dump(prediction_accuracy_clf2, f2)
            pickle.dump(prediction_accuracy_clf3, f3)