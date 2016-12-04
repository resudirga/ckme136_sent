# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 14:40:37 2016

@author: renita
"""
import os
import string,re
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold

cdir = os.path.dirname(os.path.realpath(__file__))

def get_full_reviews(input_df):
    """ Extract full reviews from input_df """
    
    df = input_df.copy()
    assert set(['SentenceId','Sentiment', 'Phrase','PhraseId']).issubset(set(df.columns))
    
    # create a new column, the number of tokens in each phrase													
    df['num_tokens'] = df['Phrase'].apply(lambda s: len(s.split()))
    df = df.ix[df.groupby('SentenceId')['num_tokens'].idxmax()]
    df.reset_index(inplace=True)
    del df['num_tokens']
    return df

def remove_stopwords(a_string):
    """ Remove stopwords from a_string """
    with open(os.path.join(cdir, 'stopwords.json'),'r') as f:
        stops = json.load(f)
    stops = [token for token in stops]    
    return " ".join([s for s in a_string.split() if s not in stops])
    
def munge(input_df):
    """ 
    Clean reviews by transforming all string to lowercase, remove punctuations, and remove stopwords.
    Input: a dataframe with 'Sentiment' (0-4) and 'Phrase' as columns     
    Return a copy of the input dataframe with a new column labeled 'Cleaned_Phrase'
    """
     
    df = input_df.copy()
    assert set(['SentenceId', 'Phrase','PhraseId']).issubset(set(df.columns))
     
    df['Cleaned_Phrase'] = df['Phrase'].apply(lambda s: s.lower())\
                                       .apply(lambda s: [tok for tok in s.split()])\
                                       .apply(lambda s: [re.sub('[' + string.punctuation + ']', ' ', tok) for tok in s])\
                                       .apply(lambda s: [re.sub('\s+', '', tok).strip() for tok in s])\
                                       .apply(lambda s: " ".join([tok for tok in s]))\
                                       .apply(lambda s: remove_stopwords(s))
    return df
    
def make_train_test(input_df, train_prop=0.8,seed=None):
    """ Divide the sentiment data into train and test dfs """
    
    df = input_df.copy()
    assert set(['SentenceId','Sentiment', 'Phrase','PhraseId']).issubset(set(df.columns))
    
    if seed: np.random.seed(seed) 
    sentids = df['SentenceId'].unique()
    mask = np.random.rand(len(sentids)) < train_prop
    train_sentids, test_sentids = sentids[mask], sentids[~mask]
    train, test = df[df['SentenceId'].isin(train_sentids)], df[df['SentenceId'].isin(test_sentids)]
    
    return train, test
    
def get_classification_accuracy(prediction, true_labels):
    """ Calculate the accuracy of prediction in %"""
    return float(sum((prediction - true_labels) == 0)) / len(true_labels) * 100
    
if __name__ == '__main__':
    
    #------------------------------------------------------    
    # Preprocess and save the cleaned data into disk
    #------------------------------------------------------
    import pickle

#    reviews = pd.read_table(cdir + '/data/train.tsv',sep='\t'
#                                          ,dtype={ 'PhraseId': str
#                                                ,'SentenceId': str
#                                                ,'Sentiment' : int 
#                                                ,'Phrase': str} )
#    
#    reviews = munge(reviews) 
#    
#    pickle.dump( reviews, open(os.path.join(cdir, 'reviews_munged.pkl'),'wb') )
    
    # Load the dataset
    reviews = pickle.load( open(os.path.join(cdir, 'reviews_munged.pkl'),'rb') )
    
    #------------------------------------------------------
    # Divide the training data into 5 (train, validation) folds and save the indices into disk   
    #------------------------------------------------------
    full_reviews = get_full_reviews(reviews)      
    train_labels = full_reviews['Sentiment']
    
    nfolds = 5
    skf = StratifiedKFold(train_labels, n_folds=nfolds,random_state=200)
    for ix_fold, (train_idx, valid_idx) in enumerate(skf): 
        train_fname = 'fold' + str(ix_fold) + '_train_SentenceId.pkl'
        valid_fname = 'fold' + str(ix_fold) + '_valid_SentenceId.pkl'
        fold_train_sentid = full_reviews.iloc[train_idx.tolist()]['SentenceId']
        fold_valid_sentid = full_reviews.iloc[valid_idx.tolist()]['SentenceId']
        
        with open(os.path.join(cdir, train_fname),'wb') as f1, open(os.path.join(cdir, valid_fname),'wb') as f2:
            pickle.dump(fold_train_sentid, f1)
            pickle.dump(fold_valid_sentid, f2)