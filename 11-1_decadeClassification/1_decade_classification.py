"""
SCRIPT 1:
This script represents the second classification that tries to find the decade of a film instead of the genres.
The same algorithm with the same parameters is used, but instead of a Multi-Label-classification with LabelPowerset only the SVC is used.

Procedure:
1. Takes feature matrix from /9_featureMatrix as input
2. Extracts the labels
3. Extracts the features and normalizes them with zscore
4. Performs classification
5. Save the results as .csv and visualizes them as .png files
"""


import glob
import os
import re
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, zscore
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

SAMPLE = "sample_1930-2010"
MATRIX_DIR = "XXX\\filmgenre_classification\\9_featureMatrix"
RESULT_DIR = "XXX\\filmgenre_classification\\11-2_results"
FEATURE_MATRIX = MATRIX_DIR + SAMPLE + "_feature_matrix.csv"

#Select matplotlib-style
plt.style.use('seaborn-whitegrid')

#Check if sample is correct:
sample = "1930-2010"

# Options: 
# "all_features"
# "only_topics"
# "only_syn"
# Make sure adjusting row/column selection in get_features_and_labels is right and the correct list in classify is selected!
used_features = "all_features"


def get_features_and_labels(FEATURE_MATRIX):
    matrix_df = pd.read_csv(FEATURE_MATRIX)
    #Get wanted features: 
    # 6: means all features, 
    # -10: means only syntactic features
    # 6:-10 means only topics
    features = matrix_df.iloc[:,6:]
    features = features.apply(zscore)
    labels = matrix_df["decade"]
    return features, labels

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
	#Function taken from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py					  
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #If default style isn't set here, there will be grid lines in the confusion matrix
    plt.style.use("default")
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def split_dataset(features, labels):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)
    return features_train, features_test, labels_train, labels_test

def classify(features_train, features_test, labels_train, labels_test):

    #Centuries as lists for usage in probability calculation. Pick or create a new list that matches the available decades
    decades = ["1930", "1940", "1950", "1960", "1970", "1980", "1990", "2000", "2010"]
    #decades = ["1970", "1980","1990", "2000", "2010"]
    #decades = ["2000", "2010"]
    
    #Set up classifier. This is the same classifier as used in the genre_classification but without LabelPowerset
    clf = SVC(C=100, gamma=0.001, cache_size=1000.0, probability=True)
    
    #Fit the classifier with training data
    clf.fit(features_train, labels_train)
    #Predict the labels of the test_set
    predicted_labels = clf.predict(features_test)
    #Save results in a dictionary, contains:
    # Precision
    # Recall
    # F1-Score 
    # Support
    # --> For each label and also micro, macro, weighted and sampled-averaged scores
    result_dict_classification_report = classification_report(labels_test, predicted_labels, target_names=decades, output_dict=True)
    
    #This confusion matrix is not yet visualized
    c_matrix = confusion_matrix(labels_test, predicted_labels)
    #print(con)
    return result_dict_classification_report, c_matrix

def visualize(dict_class_report, c_matrix):

    complete_df = pd.DataFrame.from_dict(dict_class_report, orient="index")
    #print(complete_df)
    amount_of_movies = complete_df.loc["micro avg", "support"]

    df_century_scores = complete_df.iloc[:-3,:3]
    df_century_distribution = complete_df.iloc[:-3,3:]
    
    df_century_scores.plot(kind="bar", rot=45, title="Classification results per century") 
    plt.ylabel("in %", rotation="vertical", fontsize="large")
    plt.xlabel("Centuries", fontsize="large")
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(RESULT_DIR + "\\" + sample + "_" + used_features + "_results_per_century.png", bbox_inches="tight")
    plt.show()
    
    df_century_distribution.plot(kind="bar", rot=45,  legend=False, title="Distribution of movies per century (n = " + str(amount_of_movies) + ")")
    plt.ylabel("Amount of movies", rotation="vertical", fontsize="large")
    plt.xlabel("Genres", fontsize="large")
    plt.tight_layout()
    plt.savefig(RESULT_DIR + "\\" + sample + "_" + used_features + "_century_distribution.png", bbox_inches="tight")
    plt.show()

    plt.figure()
    plot_confusion_matrix(c_matrix, classes=df_century_scores.index.values, title='Classification results')
    plt.savefig(RESULT_DIR + "\\" + sample + "_" + used_features + "_confusion_matrix.png", bbox_inches="tight")
    plt.show()
    
    class_report_df = pd.DataFrame.from_dict(dict_class_report)
    pd.DataFrame.to_csv(class_report_df, RESULT_DIR + "\\" + sample + "_" + used_features + "_classification_report.csv")

def main(FEATURE_MATRIX):
    features, labels = get_features_and_labels(FEATURE_MATRIX)
    features_train, features_test, labels_train, labels_test = split_dataset(features, labels)
    dict_class_report, c_matrix = classify(features_train, features_test, labels_train, labels_test)
    visualize(dict_class_report, c_matrix)
main(FEATURE_MATRIX)
