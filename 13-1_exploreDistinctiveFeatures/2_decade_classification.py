"""
SCRIPT 2:
This script performs another classification with the OneVsRestClassifier and LinearSVC-Kernel. This classification is needed, because the scikit-multilearn classification can NOT
return the feature coefficients that are needed to say "Feature X is important to identify Decade y"

The logic itself is the same as in the previous classification, but instead of the classification results, we are only interested in the most influential topics and their coefficients!
So only those data is saved to .csv files!
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import randint, zscore
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from collections import Counter

sample = "sample_1930-2010"
MATRIX_DIR = "XXX\\filmgenre_classification\\9_featureMatrix"
FEATURE_MATRIX = MATRIX_DIR + "\\" + sample + "_feature_matrix.csv"
RESULT_DIR = "XXX\\filmgenre_classification\\13-2_distinctiveFeatures\\decade"
        

def get_features_and_labels(FEATURE_MATRIX):
    matrix_df = pd.read_csv(FEATURE_MATRIX)
    #Get wanted features: 
    # 6: means all features, 
    # -10: means only syntactic features
    # 6:-10 means only topics
    features = matrix_df.iloc[:,6:-10]
    features = features.apply(zscore)
    labels = matrix_df["decade"]
    return features, labels

def split_dataset(features, labels):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)
    return features_train, features_test, labels_train, labels_test

def classify(features, features_train, features_test, labels_train, labels_test):
    
    decades = ["1930", "1940", "1950", "1960", "1970", "1980", "1990", "2000", "2010"]
    if sample == "1970-2010":
        decades = ["1970", "1980","1990", "2000", "2010"]
    #This sample is not working properly, because it's a binary classification problem
    if sample == "2000-2010":
        decades = ["2000", "2010"]
    
    feature_coef_dict = {}
    
    clf = OneVsRestClassifier(SVC(kernel="linear", C=100, cache_size=1000.0), n_jobs=-1)
    
    #Fit the classifier with training data
    clf.fit(features_train, labels_train)

    for estimator, decade in zip(clf.estimators_, decades):
        #There is one estimator for each class/decade, so we iterate over both
        feature_coef_dict[decade] = {}
        #Flatten the array and convert it to a list afterwards
        coef_list = estimator.coef_.ravel().tolist()
        for coe, feature in zip(coef_list, features):
            #Save the coef for each feature for each decade
            feature_coef_dict[decade][feature] = coe
    fc_df = pd.DataFrame.from_dict(feature_coef_dict)
    #print(fc_df)
    pd.DataFrame.to_csv(fc_df, RESULT_DIR + "\\" + sample + "_feature_coefficients.csv")
    
    return feature_coef_dict
    
def find_most_influential_features(feature_coef_dict):
    top_10_feature_dict = {}
    bottom_10_feature_dict = {}
    for decade, features in feature_coef_dict.items():
        #Find the 10 most influent features for classification
        top_10_feature_dict[decade] = dict(Counter(features).most_common(10))
        #Find the 10 features with least/negative influence for the classification
        bottom_10_feature_dict[decade] = dict(Counter(features).most_common()[:-10-1:-1])
    #print(top_10_feature_dict)
    return top_10_feature_dict, bottom_10_feature_dict

def save_dicts(top10_dict, bottom10_dict):

    top10_features_per_decade = {}
    for decade, features in top10_dict.items():
        list_of_topics = []
        for topic, coefs in features.items():
            list_of_topics.append(topic)
        top10_features_per_decade[decade] = list_of_topics
    top10_df = pd.DataFrame.from_dict(top10_features_per_decade)
    top10_df.set_index(np.arange(1,11,1), inplace=True)
    print(top10_df)
    pd.DataFrame.to_csv(top10_df, RESULT_DIR + "\\" + sample + "_top10_topics_per_decade.csv", sep="\t")

    bottom10_features_per_decade = {}
    for decade, features in bottom10_dict.items():
        list_of_topics = []
        for topic, coefs in features.items():
            list_of_topics.append(topic)
        bottom10_features_per_decade[decade] = list_of_topics
    bottom10_df = pd.DataFrame.from_dict(bottom10_features_per_decade)
    bottom10_df.set_index(np.arange(1,11,1), inplace=True)
    print(bottom10_df)
    pd.DataFrame.to_csv(bottom10_df, RESULT_DIR + "\\" + sample + "_bottom10_topics_per_decade.csv", sep="\t")
    
def main(FEATURE_MATRIX):
    features, labels = get_features_and_labels(FEATURE_MATRIX)
    features_train, features_test, labels_train, labels_test = split_dataset(features, labels)
    feature_coef_dict = classify(features, features_train, features_test, labels_train, labels_test)
    top10_dict, bottom10_dict = find_most_influential_features(feature_coef_dict)
    save_dicts(top10_dict, bottom10_dict)

main(FEATURE_MATRIX)