#Loading the packages
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

#Loading the data
trainset = pd.read_csv('datasets/incidents_train.csv', index_col=0)
val_data = pd.read_csv('datasets/incidents_val.csv', index_col=0)


#Defining the title pipeline for the svm
title_clf_svm = Pipeline([
    ('vect', TfidfVectorizer(strip_accents='unicode', analyzer='char', ngram_range=(3,6), max_df=0.5, min_df=3)),
     ('clf', svm.SVC(kernel='linear', C=10, class_weight='balanced')),
    ])


#Defining the train function that gets as input the pipeline and the name of the column that trains the model
def train(clf,column):
    for label in ('hazard-category', 'product-category', 'hazard', 'product'):
        print(label.upper())
        clf.fit(trainset[column], trainset[label]) # Fitiing the model in the train set
        print(f'Finished training for {label}.')
        
        # get development scores:
        val_data[label] = clf.predict(val_data[column])
        
    
train(title_clf_svm, 'title')

title_svm_pred_path = 'predictions/title_svm'

#Create the folder to save the predictions
os.makedirs(title_svm_pred_path, exist_ok=True)

#Defining the file path
file_path = os.path.join(title_svm_pred_path, 'submission.csv')

#Saving the DataFrame to the csv file
val_data.to_csv(file_path, index=False)





#Defining the text pipeline for the svm
text_clf_svm = Pipeline([
    ('vect', TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,2), max_df=0.5, min_df=5)),
     ('clf', svm.SVC(kernel='linear', C=10, class_weight='balanced')),
    ])


train(text_clf_svm, 'text')

text_svm_pred_path = 'predictions/text_svm'

#Create the folder to save the predictions
os.makedirs(text_svm_pred_path, exist_ok=True)

#Defining the file path
file_path = os.path.join(text_svm_pred_path, 'submission.csv')

#Saving the DataFrame to the csv file
val_data.to_csv(file_path, index=False)