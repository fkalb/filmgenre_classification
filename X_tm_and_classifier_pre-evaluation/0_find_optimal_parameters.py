"""

INFORMATIONAL PURPOSES ONLY!

This script is fairly chaotic and was not used a lot. It was used for testing different classifiers and comparing them to each other. But no results from it were used, because
most of the things tested, didn't work.


"""



import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import randint, zscore
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss,
                             jaccard_similarity_score, make_scorer,
                             zero_one_loss)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB, ComplementNB
#NAIVE BAYES NOT POSSIBLE WITH ZSCORE
from skmultilearn.adapt import (BRkNNaClassifier, BRkNNbClassifier,
                                MLkNN, MLTSVM)
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from skmultilearn.ensemble import RakelD, RakelO, LabelSpacePartitioningClassifier, MajorityVotingClassifier
from skmultilearn.cluster import MatrixLabelSpaceClusterer
np.set_printoptions(threshold=np.inf)

META_DIR = "D:\Masterarbeit\Daten\metadaten"

def get_features_and_labels(FEATURE_MATRIX):
    filename, extension = FEATURE_MATRIX.split(".")
    filename_parts = filename.split("_")
    print(filename_parts)
    topics = re.match("[0-9]{2,4}", filename_parts[5]).group(0)
    if len(filename_parts) == 7:
        iterations = re.match("[0-9]{2,4}", filename_parts[7]).group(0)
    else:
        iterations = "0"

    feature_df = pd.read_csv(FEATURE_MATRIX)
    #This print statement is just for output. Filenames can remain
    print("Now using", topics + " topics and", iterations + " iterations for the classification")
    #print()
    #Get wanted features
    features = feature_df.iloc[:, 5:]
    #Normalize features with zscore
    features = features.apply(zscore)
    print("Features normalised with z-scores, complete!")
    #Get the genre labels as series
    genre_series = feature_df["genres"]
    labels = create_binarized_labels(genre_series)
    print("Label binarization, complete!")
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

def test_classification(features, labels):

    names = []
    classifiers = []
    param_dists = []

    jaccard_sim = make_scorer(jaccard_similarity_score)
    f1_score_ = make_scorer(f1_score, average="micro")
    hamming_loss_ = make_scorer(hamming_loss)
    accuracy_score_ = make_scorer(accuracy_score)

    scoring = {
                "jaccard_sim" : jaccard_sim,
                "f1_score" : f1_score_, 
                "hamming_loss" : hamming_loss_,
                "accuracy score" : accuracy_score_
                }

    random_forests(names, classifiers, param_dists)
    mlpc_neural_network(names, classifiers, param_dists)
    k_nearest_neighbor_classifier(names, classifiers, param_dists)
    extra_trees(names, classifiers, param_dists)
    binary_relevance(names, classifiers, param_dists)
    label_powerset(names, classifiers, param_dists)
    #print(classifiers, param_dists)
    
    for name, classifier, param_dist in zip(names, classifiers, param_dists):
        print("Now testing:", name)
        print()
        
        grid_search = GridSearchCV(classifier,
                                   param_dist,
                                   cv=5,
                                   scoring=scoring,
                                   refit=False
                                   )
        grid_search.fit(features, labels)
        #print(grid_search.cv_results_)
        for scorer in scoring:
           report(grid_search.cv_results_, scorer)
        
def random_forests(names, classifiers, param_dists):
    names.append("RandomForest")
    classifiers.append(RandomForestClassifier(n_estimators=100, max_features=None, n_jobs=-1))
    param_dist = {
                  #"n_estimators" : [50, 100, 150, 200],
                  #"bootstrap" : [False, True],
                  #"max_features" : ["auto", None, "log2"],
                  #"class_weight" : ["balanced", "balanced_subsample", None]
                 }
    param_dists.append(param_dist)

def mlpc_neural_network(names, classifiers, param_dists):
    names.append("MPLC neural network")
    classifiers.append(MLPClassifier(max_iter=1000, alpha=1))
    param_dist = {
        #"alpha" : [0.0001, 0.001, 0.01, 1, 10, 100]
        #"alpha" : [1]
        #"hidden_layer_sizes" : [(20,20,20), (100,)]
    }
    param_dists.append(param_dist)

def k_nearest_neighbor_classifier(names, classifiers, param_dists):
    names.append("KNearestNeighbor")
    classifiers.append(KNeighborsClassifier(n_neighbors=12, weights="distance", n_jobs=-1))
    param_dist = {
                  #"n_neighbors" : range(10,15),
                  #"weights" : ["distance"]
                 }
    param_dists.append(param_dist)

def extra_trees(names, classifiers, param_dists):
    names.append("ExtraTrees")
    classifiers.append(ExtraTreesClassifier(n_estimators=180, max_features=None, n_jobs=-1))
    param_dist = {
                  #"n_estimators" : [100, 120, 140, 160, 180, 200],
                  #"max_features" : ["auto", None, "log2"],
    }
    param_dists.append(param_dist)

def br_kNN_a(names, classifiers, param_dists):
    names.append("BR_kNN_A")
    classifiers.append(BRkNNaClassifier())
    param_dist = {
        "k" : range(10,15)
    }
    param_dists.append(param_dist)

def br_kNN_b(names, classifiers, param_dists):
    names.append("BR_kNN_B")
    classifiers.append(BRkNNbClassifier())
    param_dist = {
        "k" : range(10,15)
    }
    param_dists.append(param_dist)

def ML_kNN(names, classifiers, param_dists):
    names.append("ML_kNN")
    classifiers.append(MLkNN())
    param_dist = {
        "k" : range(10,15),
        #"s" : [0.5, 0.7, 1.0]
    }
    param_dists.append(param_dist)

def binary_relevance(names, classifiers, param_dists):
    names.append("Binary Relevance")
    classifiers.append(BinaryRelevance())
    param_dist = [
    {
        "classifier" : [LinearSVC()]
    },
    {
        "classifier" : [LogisticRegression(multi_class="auto", solver="saga")]
    }
    ]
    param_dists.append(param_dist)

def classifier_chain(names, classifiers, param_dists):
    names.append("Classifier Chain")
    classifiers.append(BinaryRelevance())
    param_dist = [
    {
        "classifier" : [KNeighborsClassifier(n_neighbors=12, weights="distance")]
    },
    {
        "classifier" : [RandomForestClassifier(n_estimators=100, max_features=None)]
    },
    {
        "classifier" : [ExtraTreesClassifier(n_estimators=180, max_features=None)]
    },
    {
        "classifier" : [MLPClassifier(alpha=1, max_iter=1000)]
    },
    {
        "classifier" : [SVC()]
    },
    {
        "classifier" : [LinearSVC()]
    },
    {
        "classifier" : [LogisticRegression(multi_class="auto", solver="saga")]
    },
    {
        "classifier" : [NearestCentroid()]
    },
    {
        "classifier" : [RidgeClassifier()],
        #"classifier__class_weight" : [None, "balanced"],
        #"classifier__alpha" : [0, 0.5, 1.0]
    }
    ]
    param_dists.append(param_dist)

def label_powerset(names, classifiers, param_dists):
    names.append("Label Powerset")
    classifiers.append(LabelPowerset())
    param_dist = [
    {
        "classifier" : [SVC()]
    },
    {
        "classifier" : [LogisticRegression(multi_class="auto", solver="saga")]
        }
    ]
    param_dists.append(param_dist)

def rake_ld(names, classifiers, param_dists):
    names.append("Rake Ld")
    classifiers.append(RakelD())
    param_dist = {
        "labelset_size" : [2,3,4],
        "base_classifier" : [MLPClassifier(alpha=1, max_iter=1000), LogisticRegression(multi_class="auto", solver="saga"), SVC()]
    }
    param_dists.append(param_dist)

def rake_lo(names, classifiers, param_dists):
    names.append("Rake Lo")
    classifiers.append(RakelO())
    param_dist = {
        "base_classifier" : [MLPClassifier(alpha=1, max_iter=1000), LogisticRegression(multi_class="auto", solver="saga")],
        "model_count" : [17, 34, 51]
    }
    param_dists.append(param_dist)

""" ORIGINAL FUNCTION
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
"""

def report(results, scorer, n_top=15):
    print("Results for:", scorer)
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_' + scorer] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_' + scorer][candidate],
                  results['std_test_' + scorer][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
"""
def main(META_DIR):
    for FEATURE_MATRIX in glob.glob(META_DIR + "\sample_1930-2010_feature_matrix_*.csv"):
        features, labels = get_features_and_labels(FEATURE_MATRIX)
        test_classification(features, labels)
main(META_DIR) 
"""

#comment out main method or it will start from this one