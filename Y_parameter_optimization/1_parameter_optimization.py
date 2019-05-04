"""
SCRIPT 1:

This script was used to find the optimal parameters for the LabelPowerset Classifier from scikit-multilearn using the SVC from scikit-learn as a Base Classifier.

When testing all 676 cases with 10 fold cross validation 6760 runs, this script takes a very long time to finish depending on the size of the sample. The smallest sample took 40 hours on an Intel Xeon E3-1231 v3 Quad-Core CPU with 3,8Ghz, the largest sample would take over 400 hours to finish.
To avoid computations that take nearly one week, the parameters were only optimized for one sample and then used for the other samples as well.

The result file can be found in the same directory as this script. There is also a reduced result file, that shows only the most important data.
"""


import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import randint, zscore
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss,
                             jaccard_similarity_score, make_scorer,
                             zero_one_loss)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from skmultilearn.problem_transform import LabelPowerset

np.set_printoptions(threshold=np.inf)

META_DIR = ""

def get_features_and_labels(FEATURE_MATRIX):
    filename, extension = FEATURE_MATRIX.split(".")
    filename_parts = filename.split("_")
    sample = filename_parts[1]
    print("Now classifying sample:", sample)

    feature_df = pd.read_csv(FEATURE_MATRIX)

    #Get wanted features: 
    # 5: means all features, 
    # -10: means only syntactic features
    # 5:-10 means only topics
    features = feature_df.iloc[:, 5:-10]
    #print(features)

    #Normalize features with zscore
    features = features.apply(zscore)
    print("Features normalised with z-scores, complete!")

    #Get the genre labels as series
    genre_series = feature_df["genres"]
    labels = create_binarized_labels(genre_series)
    print("Label binarization, complete!")

    return features, labels, sample

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

def setting_up_scorers():
    scorers = {
    "jaccard_sim" : make_scorer(jaccard_similarity_score),
    "f1_micro" : make_scorer(f1_score, average="micro"),
    "f1_macro" : make_scorer(f1_score, average="macro"),
    "f1_weighted" : make_scorer(f1_score, average="weighted"),
    "f1_samples" : make_scorer(f1_score, average="samples"),
    "hamming_loss" : make_scorer(hamming_loss),
    "accuracy_percentage" : make_scorer(accuracy_score),
    "accuracy_number" : make_scorer(accuracy_score, normalize=False)
    }
    return scorers

def find_parameters(features, labels, sample):
    scorers = setting_up_scorers()
    meta_classifier = LabelPowerset(SVC(cache_size=1500.0))
    parameters = [  {
		#Test different combinations of classifiers. 676 combinations in total! = 13*13*2*2
        "classifier__C" : np.logspace(-2, 10, 13),
        "classifier__gamma" : np.logspace(-9, 3, 13),
        "classifier__class_weight" : [None, "balanced"],
        "classifier__decision_function_shape" : ["ovo", "ovr"]
    }
    ]

    grid_search = GridSearchCV(meta_classifier,
                                   parameters,
                                   cv=10,
                                   scoring=scorers,
                                   n_jobs=-1,
                                   refit=False,
                                   return_train_score=False
                                   )
    grid_search.fit(features, labels)
    result_df = pd.DataFrame.from_dict(grid_search.cv_results_)
    print(result_df)
    pd.DataFrame.to_csv(result_df, META_DIR + "\\results_" + sample + ".csv", index=False)
    #print(grid_search.cv_results_)
    #for scorer in scorers.keys():
        #report(grid_search.cv_results_, scorer, sample)

def report(results, scorer, sample, n_top=15):
    print("Results for:", scorer, "and sample", sample)
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_' + scorer] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_' + scorer][candidate],
                  results['std_test_' + scorer][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def main(META_DIR):
    for FEATURE_MATRIX in glob.glob(META_DIR + "\sample_1930-2010*.csv"):
        features, labels, sample = get_features_and_labels(FEATURE_MATRIX)
        find_parameters(features, labels, sample)
main(META_DIR) 