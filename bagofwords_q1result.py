# -*- coding: utf-8 -*-
"""
ckme136 - sentiment analysis project
Q1 - Compare the accuracy of a Linear SVC with tf-idf as input trained on:
1) all training data, i.e., including independently labeled partial phrases
2) only whole reviews 

Created on Sun Oct 30 16:55:03 2016
@author: renita
"""
import os
import numpy as np
import time
import ckme136_utl as utl
import bagofwords_models_sentdivide as bow


from   sklearn.svm import LinearSVC

cdir = os.path.dirname(os.path.realpath(__file__))
  
if __name__ == '__main__':
    
    # Writing the result into a text file
    txtfile = open('bagofwords_fullonlyVSpartial.txt', 'w')
    
    import pickle    
    
    # Load the dataset
    reviews = pickle.load( open(os.path.join(cdir, 'reviews_munged.pkl'),'rb') )
    
    num_features = 5000    
    nfolds = 5
    
    prediction_accuracy_clf1 = np.zeros((nfolds,))
    prediction_accuracy_clf2 = np.zeros((nfolds,))

    
    for ix_fold in range(nfolds):
        train_fname = 'fold' + str(ix_fold) + '_train_SentenceId.pkl'
        valid_fname = 'fold' + str(ix_fold) + '_valid_SentenceId.pkl'
        train_sentids = pickle.load( open(os.path.join(cdir, train_fname),'rb') )
        valid_sentids = pickle.load( open(os.path.join(cdir, valid_fname),'rb') )
        
        fold_train_all_df = reviews[reviews['SentenceId'].isin(train_sentids)]    # training data includes both whole and partial phrase reviews 
        fold_train_whole_df = utl.get_full_reviews(fold_train_all_df)             # training data includes only whole reviews 
        
        # Do the same thing for the validation set, but include only records with full reviews        
        fold_validation_df = reviews[reviews['SentenceId'].isin(valid_sentids)]
        fold_validation_df = utl.get_full_reviews(fold_validation_df)  
        
        # Get the tf-idf matrices        
        fold_train_all_fm, fold_all_vect = bow.get_train_feature_matrix(fold_train_all_df,tfidf=True,as_sparse=True)
        fold_train_whole_fm, fold_whole_vect = bow.get_train_feature_matrix(fold_train_whole_df,tfidf=True,as_sparse=True)
        fold_validation_fm = bow.get_test_feature_matrix(fold_validation_df, fold_all_vect,as_sparse=True)

        
        # Get the training labelslabels
        fold_train_all_labels = np.array(fold_train_all_df['Sentiment'])
        fold_train_whole_labels = np.array(fold_train_whole_df['Sentiment'])
        fold_validation_labels = np.array(fold_validation_df['Sentiment'])


        # Clf1: Linear SVC fitted on all training data, i.e., whole and partial phrases
        clf1 = LinearSVC()
        start = time.time() 
        clf1.fit(fold_train_all_fm, fold_train_all_labels)
        end = time.time()
        elapsed = end - start
        
        prediction = clf1.predict(fold_validation_fm) 
        accuracy = utl.get_classification_accuracy(prediction, fold_validation_labels)
        txtfile.write('Accuracy of Linear SVC fitted on all the training data is ' + str(accuracy) + '%\n')  
        txtfile.write('Training time in sec: ' + str(elapsed) + ' s\n\n')
        
        prediction_accuracy_clf1[ix_fold] = accuracy
        
        # Clf2: Linear SVC fitted on only whole reviews
        clf2 = LinearSVC()
        start = time.time() 
        clf2.fit(fold_train_whole_fm, fold_train_whole_labels)
        end = time.time()
        elapsed = end - start
        
        prediction = clf2.predict(fold_validation_fm) 
        accuracy = utl.get_classification_accuracy(prediction, fold_validation_labels)
        txtfile.write('Accuracy of Linear SVC fitted on only whole reviews in the training data is ' + str(accuracy) + '%\n')  
        txtfile.write('Training time in sec: ' + str(elapsed) + ' s\n\n')
        
        prediction_accuracy_clf2[ix_fold] = accuracy
        
        txtfile.write('----------------------------\n')
    
    txtfile.close()
    
    with open(os.path.join(cdir, 'acc_tfidf_validation_fitted_on_all'),'wb') as f1, open(os.path.join(cdir, 'acc_tfidf_validation_fitted_on_whole'),'wb') as f2:
            pickle.dump(prediction_accuracy_clf1, f1)
            pickle.dump(prediction_accuracy_clf2, f2)