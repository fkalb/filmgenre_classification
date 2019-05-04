"""
SCRIPT 2:

This script was used to create the result files for the different classifications. There are not only result files for each combination, but also averaged results
that were used to identify the best topic/interval combination and the best classifier. These results starting with "best_" are also included in the corresponding sample folders!
"""

import glob
import os
import re
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 8)

META_DIR = ""

average_results = {}

average_classifier_results = {
    "BR-LSVC" : {"jaccard_sim" : 0,
        "hamming_loss" : 0,
        "f1_micro" : 0,
        "f1_macro" : 0,
        "f1_weighted" : 0,
        "f1_samples" : 0,
        "accuracy_percentage" : 0,
        "accuracy_number" : 0}, 
        "BR-SVC" : {"jaccard_sim" : 0,
        "hamming_loss" : 0,
        "f1_micro" : 0,
        "f1_macro" : 0,
        "f1_weighted" : 0,
        "f1_samples" : 0,
        "accuracy_percentage" : 0,
        "accuracy_number" : 0}, 
        "ExtraTrees" : {"jaccard_sim" : 0,
        "hamming_loss" : 0,
        "f1_micro" : 0,
        "f1_macro" : 0,
        "f1_weighted" : 0,
        "f1_samples" : 0,
        "accuracy_percentage" : 0,
        "accuracy_number" : 0}, 
        "LP-LSVC" : {"jaccard_sim" : 0,
        "hamming_loss" : 0,
        "f1_micro" : 0,
        "f1_macro" : 0,
        "f1_weighted" : 0,
        "f1_samples" : 0,
        "accuracy_percentage" : 0,
        "accuracy_number" : 0}, 
        "LP-SVC" : {"jaccard_sim" : 0,
        "hamming_loss" : 0,
        "f1_micro" : 0,
        "f1_macro" : 0,
        "f1_weighted" : 0,
        "f1_samples" : 0,
        "accuracy_percentage" : 0,
        "accuracy_number" : 0}, 
        "MLPC" : {"jaccard_sim" : 0,
        "hamming_loss" : 0,
        "f1_micro" : 0,
        "f1_macro" : 0,
        "f1_weighted" : 0,
        "f1_samples" : 0,
        "accuracy_percentage" : 0,
        "accuracy_number" : 0}, 
        "RandomForest" : {"jaccard_sim" : 0,
        "hamming_loss" : 0,
        "f1_micro" : 0,
        "f1_macro" : 0,
        "f1_weighted" : 0,
        "f1_samples" : 0,
        "accuracy_percentage" : 0,
        "accuracy_number" : 0}, 
        "kNN" : {"jaccard_sim" : 0,
        "hamming_loss" : 0,
        "f1_micro" : 0,
        "f1_macro" : 0,
        "f1_weighted" : 0,
        "f1_samples" : 0,
        "accuracy_percentage" : 0,
        "accuracy_number" : 0}}

for classification_results in glob.glob(META_DIR + "\sample_genre_based_classification_*.csv"):
    result_df = pd.read_csv(classification_results, index_col=0)

    filename, extension = classification_results.split(".")
    filename_parts = filename.split("_")
    print(filename_parts)
    #Put 4 and 5 as parts for decade based sample. 6 and 7 for genre_based sample
    topics = re.match("[0-9]{2,3}t", filename_parts[6]).group(0)
    iterations = re.match("[0-9]{2,4}i", filename_parts[7]).group(0)
    topic_iterations = topics + "_" + iterations

    average_results[topic_iterations] = {
        "jaccard_sim" : result_df["jaccard_sim"].mean(),
        "hamming_loss" : result_df["hamming_loss"].mean(),
        "f1_micro" : result_df["f1_micro"].mean(),
        "f1_macro" : result_df["f1_macro"].mean(),
        "f1_weighted" : result_df["f1_weighted"].mean(),
        "f1_samples" : result_df["f1_samples"].mean(),
        "accuracy_percentage" : result_df["accuracy_percentage"].mean(),
        "accuracy_number" : result_df["accuracy_number"].mean()
    }
    
    for classifier_row in result_df.itertuples():
        average_classifier_results[classifier_row.Index]["jaccard_sim"] += classifier_row.jaccard_sim
        average_classifier_results[classifier_row.Index]["hamming_loss"] += classifier_row.hamming_loss
        average_classifier_results[classifier_row.Index]["f1_micro"] += classifier_row.f1_micro
        average_classifier_results[classifier_row.Index]["f1_macro"] += classifier_row.f1_macro
        average_classifier_results[classifier_row.Index]["f1_weighted"] += classifier_row.f1_weighted
        average_classifier_results[classifier_row.Index]["f1_samples"] += classifier_row.f1_samples
        average_classifier_results[classifier_row.Index]["accuracy_percentage"] += classifier_row.accuracy_percentage
        average_classifier_results[classifier_row.Index]["accuracy_number"] += classifier_row.accuracy_number


for inner_dict in average_classifier_results.values():
    for metric in inner_dict:
        #Divide every value by 36 to get the average
        inner_dict[metric] /= 36

#print(average_classifier_results)   

average_df = pd.DataFrame.from_dict(average_results, orient="index")
average_classifier_df = pd.DataFrame.from_dict(average_classifier_results, orient="index")

print(average_df)
print()
print(average_classifier_df)

pd.DataFrame.to_csv(average_df, META_DIR + "\sample_genre_based_classification_best_topic_combination.csv")
pd.DataFrame.to_csv(average_classifier_df, META_DIR + "\sample_genre_based_classification_best_classifier_performance.csv")