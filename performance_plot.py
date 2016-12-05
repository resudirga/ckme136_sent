# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 19:51:29 2016

@author: renita
"""
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

cdir = os.path.dirname(os.path.realpath(__file__))

#-----------------------------
# Bag of Words (Tf-Idf)
#-----------------------------
num_features = ['all', '5000', '2500', '1250', '700', '500']
accuracy_nb = pickle.load(open(os.path.join(cdir,'tfidf_5folds_nb_acc'),'rb'))
accuracy_svc = pickle.load(open(os.path.join(cdir,'tfidf_5folds_svc_validation_acc'),'rb'))
accuracy_forest = pickle.load(open(os.path.join(cdir,'tfidf_5folds_forest_validation_acc'),'rb'))

mean_accuracy_nb = np.mean(accuracy_nb,axis=0)
stdev_accuracy_nb = np.std(accuracy_nb,axis=0)
fig, ax = plt.subplots()
xtick_pos = np.arange(0,len(num_features))
xticklabels = [num for num in num_features]
ax.set_xticks(xtick_pos)
ax.set_xticklabels(xticklabels, rotation=0)
plt.ylabel('Prediction accuracy (%)')
plt.xlabel('Size of feature vectors')
plt.xlim([-.1, 5.2])
plt.ylim([45, 60])
plt.plot(np.arange(0,len(num_features)),mean_accuracy_nb, 'r', label='Naive Bayes')
plt.errorbar(xtick_pos, mean_accuracy_nb, yerr=stdev_accuracy_nb,fmt='rd')


mean_accuracy_svc = np.mean(accuracy_svc,axis=0)
stdev_accuracy_svc = np.std(accuracy_svc,axis=0)
plt.plot(np.arange(0,len(num_features)),mean_accuracy_svc, 'm--', label='SVC')
plt.errorbar(xtick_pos, mean_accuracy_svc, yerr=stdev_accuracy_svc,fmt='m+')

mean_accuracy_forest = np.mean(accuracy_forest,axis=0)
stdev_accuracy_forest =  np.std(accuracy_forest,axis=0)
plt.plot(np.arange(0,len(num_features)),mean_accuracy_forest, 'k-.', label='Random Forest (ntrees=100)')
plt.errorbar(xtick_pos, mean_accuracy_forest, yerr=stdev_accuracy_forest,fmt='ko')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.title('Classifier accuracy with bag of words as features (5 folds)', y=1.2, fontsize = 14)
           
fig.savefig(os.path.join(cdir,"TfIdf_validation_accuracy.png"), bbox_inches='tight', dpi=300)

#-----------------------------
# Word2Vec
#-----------------------------

# a) Word2Vec model trained only on the training data
num_features = ['100', '200', '300', '500', '1000']
accuracy_svc = pickle.load(open(os.path.join(cdir,'w2v_5folds_svc_validation_acc'),'rb'))
accuracy_forest = pickle.load(open(os.path.join(cdir,'w2v_5folds_forest_validation_acc'),'rb'))


fig, ax = plt.subplots()
xtick_pos = np.arange(0,len(num_features))
xticklabels = [num for num in num_features]
ax.set_xticks(xtick_pos)
ax.set_xticklabels(xticklabels, rotation=0)
plt.ylabel('Prediction accuracy (%)')
plt.xlabel('Number of words')
plt.xlim([-.1, 4.2])
plt.ylim([45, 60])
mean_accuracy_svc = np.mean(accuracy_svc,axis=0)
stdev_accuracy_svc = np.std(accuracy_svc,axis=0)
plt.plot(np.arange(0,len(num_features)),mean_accuracy_svc, 'm--', label='SVC')
plt.errorbar(xtick_pos, mean_accuracy_svc, yerr=stdev_accuracy_svc,fmt='m+')

mean_accuracy_forest = np.mean(accuracy_forest,axis=0)
stdev_accuracy_forest = np.std(accuracy_forest,axis=0)

plt.plot(np.arange(0,len(num_features)),mean_accuracy_forest, 'k--', label='Random Forest')
plt.errorbar(xtick_pos, mean_accuracy_forest, yerr=stdev_accuracy_forest,fmt='ko')


# b) Word2Vec model trained on 233k IMDB movie review data. Only 3 folds were evaluated
accuracy_svc = pickle.load(open(os.path.join(cdir,'w2v_sar14_5folds_svc_validation_acc'),'rb'))
accuracy_forest = pickle.load(open(os.path.join(cdir,'w2v_sar14_5folds_forest_validation_acc'),'rb'))
mean_accuracy_svc = np.mean(accuracy_svc[0:3, :],axis=0)
stdev_accuracy_svc = np.std(accuracy_svc[0:3, :],axis=0)
plt.plot(np.arange(0,len(num_features)),mean_accuracy_svc, 'm-', label='SVC, IMDB')
plt.errorbar(xtick_pos, mean_accuracy_svc, yerr=stdev_accuracy_svc,fmt='m+')

mean_accuracy_forest = np.mean(accuracy_forest[0:3, :],axis=0)
stdev_accuracy_forest = np.std(accuracy_forest[0:3, :],axis=0)
plt.plot(np.arange(0,len(num_features)),mean_accuracy_forest, 'k-', label='Random Forest, IMDB')
plt.errorbar(xtick_pos, mean_accuracy_forest, yerr=stdev_accuracy_forest,fmt='k+')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.title('Classifier accuracy with Word2Vec vectors as features (5 folds)', y=1.2, fontsize = 14)
           
fig.savefig(os.path.join(cdir,"Word2Vec_validation_accuracy.png"), bbox_inches='tight', dpi=300)
#-----------------------------
# SO-CAL
#-----------------------------
num_features = ['sum', '2', '5', '10']
accuracy_svc = pickle.load(open(os.path.join(cdir,'socal_5folds_svc_validation_acc'),'rb'))
accuracy_forest10 = pickle.load(open(os.path.join(cdir,'socal_5folds_forest10_validation_acc'),'rb'))
accuracy_forest50 = pickle.load(open(os.path.join(cdir,'socal_5folds_forest50_validation_acc'),'rb'))

fig, ax = plt.subplots()
xtick_pos = np.arange(0,len(num_features))
xticklabels = [num for num in num_features]
ax.set_xticks(xtick_pos)
ax.set_xticklabels(xticklabels, rotation=0)
plt.ylabel('Prediction accuracy (%)')
plt.xlabel('Number of words')
plt.xlim([-.1, 3.2])
plt.ylim([45, 60])

mean_accuracy_svc = np.mean(accuracy_svc,axis=0)
stdev_accuracy_svc = np.std(accuracy_svc,axis=0)
plt.plot(np.arange(0,len(num_features)),mean_accuracy_svc, 'm--', label='SVC')
plt.errorbar(xtick_pos, mean_accuracy_svc, yerr=stdev_accuracy_svc,fmt='m+')

mean_accuracy_forest10 = np.mean(accuracy_forest10,axis=0)
stdev_accuracy_forest10 = np.std(accuracy_forest10,axis=0)
plt.plot(np.arange(0,len(num_features)),mean_accuracy_forest10, 'k-.', label='Random Forest (ntrees=10)')
plt.errorbar(xtick_pos, mean_accuracy_forest10, yerr=stdev_accuracy_forest10,fmt='ko')

mean_accuracy_forest50 = np.mean(accuracy_forest50,axis=0)
stdev_accuracy_forest50 = np.std(accuracy_forest50,axis=0)
plt.plot(np.arange(0,len(num_features)),mean_accuracy_forest50, 'b:', label='Random Forest (ntrees=50)')
plt.errorbar(xtick_pos, mean_accuracy_forest50, yerr=stdev_accuracy_forest50,fmt='bd')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.title('Classifier accuracy with SO-CAL word-score ngram as features (5 folds)', y=1.2, fontsize = 14)
           
fig.savefig(os.path.join(cdir,"SOCAL_validation_accuracy.png"), bbox_inches='tight', dpi=300)