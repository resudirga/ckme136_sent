# -*- coding: utf-8 -*-
"""
ckme136 - sentiment analysis project
part 1a, bag-of-words models
Created on Sun Oct 30 16:55:03 2016
@author: renita
"""
import os
import numpy as np
import time
import ckme136_utl as utl
from   sklearn.feature_extraction.text import CountVectorizer
from   sklearn.feature_extraction.text import TfidfVectorizer

from   sklearn.naive_bayes import MultinomialNB
from   sklearn.svm import LinearSVC
from   sklearn.linear_model import SGDClassifier
from   sklearn.multiclass import OneVsRestClassifier 
from   sklearn.ensemble import RandomForestClassifier

cdir = os.path.dirname(os.path.realpath(__file__))

    
def get_train_feature_matrix(df,tfidf=True,vocabulary=None,as_sparse=False,max_features=None):
    """ Given an input df, return features matrix and the corresponding vectorizer """
    assert set(['SentenceId','Sentiment', 'Phrase','PhraseId', 'Cleaned_Phrase']).issubset(set(df.columns))
    
    if tfidf:
        if vocabulary: 
            vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b', vocabulary=vocabulary)
        else: 
            vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b',max_features=max_features)
    else: 
        if vocabulary: 
            vectorizer = CountVectorizer(token_pattern=r'\b\w+\b', vocabulary=vocabulary)
        else: 
            vectorizer = CountVectorizer(token_pattern=r'\b\w+\b',max_features=max_features)
    
    feature_matrix = vectorizer.fit_transform(df['Cleaned_Phrase'])
    if not as_sparse: 
        feature_matrix = feature_matrix.toarray()
        
    return feature_matrix, vectorizer
    
def get_test_feature_matrix(df,vectorizer,as_sparse=False):
    """ Given an input df and a vectorizer return the corresponding features matrix"""
    assert set(['Cleaned_Phrase']).issubset(set(df.columns))
    feature_matrix = vectorizer.transform(df['Cleaned_Phrase'])
    if not as_sparse: 
        feature_matrix = feature_matrix.toarray()
    return feature_matrix
    
    
if __name__ == '__main__':
    
    # Writing the result into a text file
    txtfile = open('bagofwords_performance_sentdivide_folds_featsel.txt', 'w')
    
    import pickle    
    
    # Load the dataset
    reviews = pickle.load( open(os.path.join(cdir, 'reviews_munged.pkl'),'rb') )
    
    num_features_opts = [5000, 2500, 1250, 700, 500]
    
    nfolds = 5
    
    prediction_accuracy_clf1 = np.zeros([nfolds, len(num_features_opts) + 1])
    prediction_accuracy_clf2 = np.zeros([nfolds, len(num_features_opts) + 1])
    prediction_accuracy_clf3 = np.zeros([nfolds, len(num_features_opts) + 1])
    
    # ========================================================
    # Part 1: Using Td-Idf of all words in the training set as features
    # ========================================================
    txtfile.write('====== Part 1: Using Tf-Idf of all words in the training set as features =====\n')    
    
    
    for ix_fold in range(nfolds):
        train_fname = 'fold' + str(ix_fold) + '_train_SentenceId.pkl'
        valid_fname = 'fold' + str(ix_fold) + '_valid_SentenceId.pkl'
        train_sentids = pickle.load( open(os.path.join(cdir, train_fname),'rb') )
        valid_sentids = pickle.load( open(os.path.join(cdir, valid_fname),'rb') )
        
        fold_train_df = reviews[reviews['SentenceId'].isin(train_sentids)]
        fold_validation_df = reviews[reviews['SentenceId'].isin(valid_sentids)]
        
        fold_train_fm, fold_vect = get_train_feature_matrix(fold_train_df,tfidf=True,as_sparse=True)
        fold_train_labels = np.array(fold_train_df['Sentiment'])
        
        fold_validation_fm = get_test_feature_matrix(fold_validation_df, fold_vect,as_sparse=True)
        validation_labels = np.array(fold_validation_df['Sentiment'])
    
    #    # Clf1: Gaussian Naive Bayes classifier with uniform prior
    #    clf1 = GaussianNB()
    #    start = time.time()    
    #    clf1.fit(train_feature_matrix.toarray(), train_labels)
    #    end = time.time()
    #    elapsed = end - start
    #    prediction = clf1.predict(test_feature_matrix.toarray()) 
    #    accuracy = utl.get_classification_accuracy(prediction, test_labels)
    #    txtfile.write('Accuracy of Gaussian NB classifier is ' + str(accuracy) + '%\n')    
    #    txtfile.write('Training time in sec: ' + str(elapsed) + ' s\n\n')
        
        # Clf2: Multinomial Naive Bayes classifier with uniform prior
        clf2 = MultinomialNB()
        start = time.time() 
        clf2.fit(fold_train_fm, fold_train_labels)
        end = time.time()
        elapsed = end - start
        
        prediction = clf2.predict(fold_validation_fm) 
        accuracy = utl.get_classification_accuracy(prediction, validation_labels)
        txtfile.write('Accuracy of Multinomial NB classifier is ' + str(accuracy) + '%\n')  
        txtfile.write('Training time in sec: ' + str(elapsed) + ' s\n\n')
        
        prediction_accuracy_clf1[ix_fold, 0] = accuracy
        
        # Clf3: Linear SVC
        clf3 = LinearSVC()
        start = time.time() 
        clf3.fit(fold_train_fm, fold_train_labels)
        end = time.time()
        elapsed = end - start
        
        prediction = clf3.predict(fold_validation_fm) 
        accuracy = utl.get_classification_accuracy(prediction, validation_labels)
        txtfile.write('Accuracy of Linear SVC is ' + str(accuracy) + '%\n')  
        txtfile.write('Training time in sec: ' + str(elapsed) + ' s\n\n')
        
        prediction_accuracy_clf2[ix_fold, 0] = accuracy
        
        # Clf4: SGD Classifier with OnevsRest
        clf4 = OneVsRestClassifier(SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001))
        start = time.time() 
        clf4.fit(fold_train_fm, fold_train_labels)
        end = time.time()
        elapsed = end - start
        
        prediction = clf4.predict(fold_validation_fm) 
        accuracy = utl.get_classification_accuracy(prediction, validation_labels)
        txtfile.write('Accuracy of SGD Classifier with OneVsRest is ' + str(accuracy) + '%\n')   
        txtfile.write('Training time in sec: ' + str(elapsed) + ' s\n\n')
        
        # Clf5: Random Forest classifier
        clf5 = RandomForestClassifier(n_estimators=100,n_jobs=-1,min_samples_leaf=10)
        start = time.time() 
        clf5.fit(fold_train_fm, fold_train_labels)
        end = time.time()
        
        prediction = clf5.predict(fold_validation_fm) 
        accuracy = utl.get_classification_accuracy(prediction, validation_labels)
        txtfile.write('Accuracy of Random Forest classifier (ntree=100) is ' + str(accuracy) + '%\n')
        txtfile.write('Training time in sec: ' + str(elapsed) + ' s\n\n')
        
        prediction_accuracy_clf3[ix_fold, 0] = accuracy
    
    # ========================================================
    # Part 2: Using Td-Idf as features, limiting the number of words 
    # included in the feature set by their term frequencies
    # ========================================================
    txtfile.write('====== Part 2: Limiting the number of words in the feature set =====\n') 
    
    
    nfolds = 5
    
    
    txtfile.write('Applying a %d-fold cross validation: \n\n' %nfolds) 
    
    for ix_feature, max_features in enumerate(num_features_opts):
        txtfile.write('Max Features = %s' %max_features + '\n')
        print('Max Features = %s' %max_features + '\n')
        txtfile.write('----------------------------\n')
        
        for ix_fold in range(nfolds):
            train_fname = 'fold' + str(ix_fold) + '_train_SentenceId.pkl'
            valid_fname = 'fold' + str(ix_fold) + '_valid_SentenceId.pkl'
            train_sentids = pickle.load( open(os.path.join(cdir, train_fname),'rb') )
            valid_sentids = pickle.load( open(os.path.join(cdir, valid_fname),'rb') )
            
            fold_train_df = reviews[reviews['SentenceId'].isin(train_sentids)]
            fold_validation_df = reviews[reviews['SentenceId'].isin(valid_sentids)]
            
            fold_train_fm, fold_vect = get_train_feature_matrix(fold_train_df,tfidf=True, max_features=max_features,as_sparse=True)
            fold_train_labels = np.array(fold_train_df['Sentiment'])
            
            fold_validation_fm = get_test_feature_matrix(fold_validation_df, fold_vect,as_sparse=True)
            validation_labels = np.array(fold_validation_df['Sentiment'])
            
            # Clf1: Multinomial NB
            fold_clf1 = MultinomialNB()
            fold_clf1.fit(fold_train_fm, fold_train_labels)
            
            fold_prediction = fold_clf1.predict(fold_validation_fm) 
            accuracy_fold_prediction = utl.get_classification_accuracy(fold_prediction, validation_labels)
            
            prediction_accuracy_clf1[ix_fold, ix_feature + 1] = accuracy_fold_prediction
            txtfile.write('Fold: %d - Number of vocabularies = %d\n' %(ix_fold, len(fold_vect.vocabulary_)) )
            txtfile.write('Accuracy of Multinomial NB classifier is %s\n' %accuracy_fold_prediction) 
            print('Accuracy of Multinomial NB classifier is %s\n' %accuracy_fold_prediction) 
            
            del fold_clf1
            
            # Clf2: Linear SVC
            fold_clf2 = LinearSVC()
            fold_clf2.fit(fold_train_fm, fold_train_labels)
            
            fold_prediction = fold_clf2.predict(fold_validation_fm) 
            accuracy_fold_prediction = utl.get_classification_accuracy(fold_prediction, validation_labels)
            
            prediction_accuracy_clf2[ix_fold, ix_feature + 1] = accuracy_fold_prediction
            txtfile.write('Accuracy of Linear SVC is %s\n' %accuracy_fold_prediction) 
            print('Accuracy of Linear SVC is %s\n' %accuracy_fold_prediction)
            
            del fold_clf2
            
            # Clf3: Random Forest
            fold_clf3 = RandomForestClassifier(n_estimators=100,n_jobs=-1,min_samples_leaf=10)
            fold_clf3.fit(fold_train_fm, fold_train_labels)
            
            fold_prediction = fold_clf3.predict(fold_validation_fm) 
            accuracy_fold_prediction = utl.get_classification_accuracy(fold_prediction, validation_labels)
            
            prediction_accuracy_clf3[ix_fold, ix_feature + 1] = accuracy_fold_prediction
            txtfile.write('Accuracy of Random Forest is %s\n' %accuracy_fold_prediction) 
            print('Accuracy of Random Forest is %s\n' %accuracy_fold_prediction) 
            
            del fold_clf3
            
            txtfile.write('----------------------------\n')
    
    txtfile.close()
    
    with open(os.path.join(cdir, 'tfidf_5folds_nb_acc'),'wb') as f1, open(os.path.join(cdir, 'tfidf_5folds_svc_validation_acc'),'wb') as f2, open(os.path.join(cdir, 'tfidf_5folds_forest_validation_acc'),'wb') as f3:
            pickle.dump(prediction_accuracy_clf1, f1)
            pickle.dump(prediction_accuracy_clf2, f2)
            pickle.dump(prediction_accuracy_clf3, f3)