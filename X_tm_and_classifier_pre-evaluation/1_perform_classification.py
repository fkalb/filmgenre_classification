"""
SCRIPT 1:
This script was used to test many different classifiers and topic/interval combinations in one run. This script can take several
days to finish, even on a fast computer, so it was not included into the pipeline (Folders 0 to 14-2). The results were also analyzed manually.

The feature matrices used for that purpose are too large in total (About 2GB in total, 144 combinations were tested), so they can't be found in the repository. 

The classification results from that process are uploaded
and can be found in the corresponding sample folder.
"""


import glob
import os
import re


import numpy as np
import pandas as pd
from scipy.stats import randint, zscore
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss,
                             jaccard_similarity_score, make_scorer,
                             zero_one_loss)
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC, LinearSVC
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset


META_DIR = ""

def get_features_and_labels(FEATURE_MATRIX):
    filename, extension = FEATURE_MATRIX.split(".")
    filename_parts = filename.split("_")
    print(filename_parts)
    #Change parts to 7 when using genre_based. Default values are: Parts 5 and 6, len = 7
    topics = re.match("[0-9]{2,4}", filename_parts[5]).group(0)
    #Change len to 9 and parts to 8, when using genre_based
    if len(filename_parts) == 7:
        iterations = re.match("[0-9]{2,4}", filename_parts[6]).group(0)
    else:
        print("Iteration error!")

    feature_df = pd.read_csv(FEATURE_MATRIX)
    #This print statement is just for output. Filenames can remain
    print("Now using", topics + " topics and", iterations + " iterations for the classification")
    #Get wanted features
    features = feature_df.iloc[:, 5:]
    #Normalize features with zscore
    features = features.apply(zscore)
    #Get the genre labels as series
    genre_series = feature_df["genres"]
    #Binarize the labels
    labels = create_binarized_labels(genre_series)
    return features, labels, topics, iterations

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
    
def write_results(number_of_topics, number_of_iterations, result_dict):
    topics = number_of_topics + "t"
    iterations = number_of_iterations + "i"
    topiter = topics + "_" + iterations

    result_df = pd.DataFrame.from_dict(result_dict, orient="index")
    #print(result_df)
    pd.DataFrame.to_csv(result_df, META_DIR + "\\sample_2000-2010_classification_" + topiter + ".csv")

def main(META_DIR):
    
    scoring_metrics = setting_up_scorers()

    for FEATURE_MATRIX in glob.glob(META_DIR + "\sample_2000-2010_feature_matrix_70t_*.csv"):
        features, labels, number_of_topics, number_of_iterations = get_features_and_labels(FEATURE_MATRIX)
        import warnings
        warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
        clf_names = ["ExtraTrees", "RandomForest", "kNN", "MLPC", "BR-SVC", "BR-LSVC", "LP-SVC", "LP-LSVC"]
        classifiers = [ExtraTreesClassifier(n_estimators=100),
            RandomForestClassifier(n_estimators=100),
            KNeighborsClassifier(n_neighbors=5),
            MLPClassifier(max_iter=1500),
            BinaryRelevance(classifier=SVC(gamma="auto")),
            BinaryRelevance(classifier=LinearSVC(dual=False, max_iter=1500)),
            LabelPowerset(classifier=SVC(gamma="auto")),
            LabelPowerset(classifier=LinearSVC(dual=False, max_iter=1500))
                    ]
        #Overall results that contain every classifier
        results = {}
        for name, clf in zip(clf_names, classifiers):
            #Dict for interim results, gets filled with another classifier's results in each iteration
            metric_results = {}
            scores = cross_validate(clf, features, labels, scoring=scoring_metrics, cv=10, return_train_score=False, n_jobs=4)
            #print(scores)
            for metric in scoring_metrics:
                #Calculate the mean of all CV scores!
                metric_results[metric] = scores["test_" + metric].mean()
            results[name] = metric_results
            #print(results)
        write_results(number_of_topics, number_of_iterations, results)
        print(number_of_topics, "and", number_of_iterations, "result file created!")
    print("All tasks completed!")
main(META_DIR) 
