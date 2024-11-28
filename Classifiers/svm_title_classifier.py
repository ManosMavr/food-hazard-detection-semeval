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


    #Defining the title pipeline for the svm
    title_clf_svm = Pipeline([
        ('vect', TfidfVectorizer(strip_accents='unicode', analyzer='char', ngram_range=(3,6), max_df=0.5, min_df=3)),
        ('clf', svm.SVC(kernel='linear', C=10, class_weight='balanced')),
        ])


    #Training the title model
    print('Training the SVM title model.')
    train(title_clf_svm, 'title')

    #Path to save the predictions
    title_svm_pred_path = 'Predictions/title_svm'

    #Create the folder to save the predictions
    os.makedirs(title_svm_pred_path, exist_ok=True)

    #Defining the file path
    title_file_path = os.path.join(title_svm_pred_path, 'submission.csv')

    #Saving the DataFrame to the csv file
    val_data.to_csv(title_file_path, index=False)


