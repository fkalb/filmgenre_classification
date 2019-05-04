"""
SCRIPT 1:
This script is the centerpiece of the pipeline as the (genre) classification part happens in here.
Procedure:
1. Takes a feature_matrix from /9_featureMatrix as input.
2. Extracts the features and normalizes them with zscore.
3. Extracts and binarizes the extracted multiple labels per movie.
3. Performs a classification with LabelPowerset and SVC.
4. Save the results as .csv files and visualizes the results as .png files.
"""


import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, zscore
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, coverage_error,
                             hamming_loss, jaccard_similarity_score,
                             label_ranking_average_precision_score,
                             label_ranking_loss)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from skmultilearn.problem_transform import LabelPowerset

SAMPLE = "sample_1930-2010"
MATRIX_DIR = "XXX\\filmgenre_classification\\9_featureMatrix"
FEATURE_MATRIX = MATRIX_DIR + "\\" + SAMPLE + "_feature_matrix.csv")
RESULT_DIR = "XXX\\filmgenre_classification\\10-2_results\\" + SAMPLE

#Select matplotlib-style
plt.style.use('seaborn-whitegrid')

#Check if sample is correct:
sample = "1930-2010"
# Options: 
# "all_features"
# "only_topics"
# "only_syn"
# Make sure adjusting row/column selection in get_features_and_labels is right!
used_features = "all_features"

def get_features_and_labels(FEATURE_MATRIX):
    matrix_df = pd.read_csv(FEATURE_MATRIX)
    #Get wanted features: 
    # 6: means all features, 
    # -10: means only syntactic features
    # 6:-10 means only topics
    features = matrix_df.iloc[:,6:]
    features = features.apply(zscore)
    labels = create_binarized_labels(matrix_df["genres"])
    return features, labels

def create_binarized_labels(genres):
    #Genres need to be lists of lists and separated by commas
    genres_with_sep = []
    for genre in list(genres):
        genres_with_sep.append(genre.split(" "))

    #Create a two-dimensional array with 1/0s for occurrence/absence of genre
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(genres_with_sep)

    #Check correct size of array, should be: "number of movies * unique genres"
    #print(labels.size)

    #Check unique genres:
    #print(mlb.classes_)

    return labels

def split_dataset(features, labels):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)
    return features_train, features_test, labels_train, labels_test

def classify(features_train, features_test, labels_train, labels_test):

    #Genres as lists for usage in probability calculation:
    genres = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Family", "Fantasy", "History", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    #Use this list only when using the sample without Drama and Comedy:
    #genres = ["Action", "Adventure", "Animation", "Crime", "Family", "Fantasy", "History", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    
    #Set up classifier
    clf = LabelPowerset(SVC(C=100, gamma=0.001, cache_size=1000.0, probability=True))
    
    #Fit the classifier with training data
    clf.fit(features_train, labels_train)
    #Predict the labels of the test_set
    predicted_labels = clf.predict(features_test)
    #Calculate the class_probabilities for each sample and convert to dense array with shape (n_samples, n_labels)
    class_probabilities = clf.predict_proba(features_test).toarray()
    #Save results in a dictionary, contains:
    # Precision
    # Recall
    # F1-Score 
    # Support
    # --> For each label and also micro, macro, weighted and sampled-averaged scores
    result_dict_classification_report = classification_report(labels_test, predicted_labels, target_names=genres, output_dict=True)
    #print(result_dict_classification_report)

    result_dict_example_based = {
        "accuracy" : clf.score(features_test, labels_test),
        "jaccard_sim" : jaccard_similarity_score(labels_test, predicted_labels),
        "average_precision_score" : average_precision_score(labels_test, class_probabilities),
        "hamming_loss" : hamming_loss(labels_test, predicted_labels)
    }
    #print(result_dict_example_based)

    result_dict_label_based = {
        #Label cardinality shows the average number of labels per sample
        #Label density puts the total number of labels into the calculation
        "label_ranking_average_precision_score" : label_ranking_average_precision_score(labels_test, class_probabilities),
        "label_cardinality" : result_dict_classification_report["micro avg"]["support"] / labels_test.shape[0],
        "label_density" : (result_dict_classification_report["micro avg"]["support"] / labels_test.shape[0]) / labels_test.shape[1],
        "label_ranking_loss" : label_ranking_loss(labels_test, class_probabilities),
        "coverage_error" : coverage_error(labels_test, class_probabilities)
    }
    #print(result_dict_label_based)

    return result_dict_classification_report, result_dict_example_based, result_dict_label_based

def save_results_as_csv(dict_class_report, dict_example_based, dict_label_based):
    class_report_df = pd.DataFrame.from_dict(dict_class_report)
    pd.DataFrame.to_csv(class_report_df, RESULT_DIR + "\\" + sample + "_" + used_features + "_classification_report.csv")
    example_based_df = pd.DataFrame.from_dict(dict_example_based, orient="index")
    pd.DataFrame.to_csv(example_based_df, RESULT_DIR + "\\" + sample +  "_" + used_features + "_example_based_results.csv")
    label_based_df = pd.DataFrame.from_dict(dict_label_based, orient="index")
    pd.DataFrame.to_csv(label_based_df, RESULT_DIR + "\\" + sample +  "_" + used_features + "_label_based_results.csv")

def visualize(dict_class_report):

    complete_df = pd.DataFrame.from_dict(dict_class_report, orient="index")
    
    #This just extracts the amount of movies from one column where averaging was done. Could also be "macro avg" or "weighted avg"
    amount_of_movies = complete_df.loc["micro avg", "support"]

    df_genre_scores = complete_df.iloc[:-4,:3]
    df_genre_distribution = complete_df.iloc[:-4,3:]
    df_overall_scores = complete_df.iloc[-4:,:3]
    
    
    df_genre_scores.plot(kind="bar", rot=45, title="Classification results per genre", figsize=(5.5,4)) 
    plt.ylabel("in %", rotation="vertical", fontsize="large")
    plt.xlabel("Genres", fontsize="large")
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(RESULT_DIR + "\\" + sample + "_" + used_features + "_results_per_genre.png", bbox_inches="tight")
    plt.show()
    
    df_overall_scores.plot(kind="bar", rot=45, title="Overall classification results with different averaging", figsize=(5.5,4)) 
    plt.ylabel("in %", rotation="vertical", fontsize="large")
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(RESULT_DIR + "\\" + sample + "_" + used_features + "_results_overall.png", bbox_inches="tight")
    plt.show()
    
    df_genre_distribution.plot(kind="bar", rot=45,  legend=False, title="Distribution of movies per genre (n = " + str(amount_of_movies) + ")", figsize=(5.5,4))
    plt.ylabel("Amount of movies", rotation="vertical", fontsize="large")
    plt.xlabel("Genres", fontsize="large")
    plt.tight_layout()
    plt.savefig(RESULT_DIR + "\\" + sample + "_" + used_features + "_genre_distribution.png", bbox_inches="tight")
    plt.show()

def main(FEATURE_MATRIX):
    features, labels = get_features_and_labels(FEATURE_MATRIX)
    features_train, features_test, labels_train, labels_test = split_dataset(features, labels)
    dict_class_report, dict_example_based, dict_label_based = classify(features_train, features_test, labels_train, labels_test)
    visualize(dict_class_report)
    save_results_as_csv(dict_class_report, dict_example_based, dict_label_based)
main(FEATURE_MATRIX)
