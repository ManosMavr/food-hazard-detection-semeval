#Loading the packages
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

#Defining the train function that gets as input the pipeline and the name of the column that trains the model
def train(clf,column):
    for label in ('hazard-category', 'product-category', 'hazard', 'product'):
        print(label.upper())
        clf.fit(trainset[column], trainset[label]) # Fitiing the model in the train set
        print(f'Finished training for {label}.')
        
        # get development scores:
        val_data[label] = clf.predict(val_data[column])


        
if __name__ == '__main__':
    #Loading the data
    trainset = pd.read_csv('Datasets/cleaned_train.csv', index_col=0)
    val_data = pd.read_csv('Datasets/cleaned_val.csv', index_col=0)


    #Defining the text pipeline for the svm
    text_clf_svm = Pipeline([
        ('vect', TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,2), max_df=0.5, min_df=5)),
        ('clf', svm.SVC(kernel='linear', C=10, class_weight='balanced')),
        ])


    #Training the text model
    print('Training the SVM text model.')
    train(text_clf_svm, 'text')

    #Path to save the predictions
    text_svm_pred_path = 'Predictions/text_svm'

    #Create the folder to save the predictions
    os.makedirs(text_svm_pred_path, exist_ok=True)

    #Defining the file path
    text_file_path = os.path.join(text_svm_pred_path, 'submission.csv')

    #Saving the DataFrame to the csv file
    val_data.to_csv(text_file_path, index=False)