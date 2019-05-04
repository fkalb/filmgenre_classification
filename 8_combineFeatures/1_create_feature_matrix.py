"""
SCRIPT 1:
This script takes the output files from Mallet and changes them into a more useful format.
After that conversion they are combined with the syntactic-features-file into a feature matrix with the corresponding metadata for each movie.
"""

import pandas as pd
import os
import glob
import re
import itertools

SAMPLE = "sample_1930-2010"
SAMPLE_DIR = "XXX\\filmgenre_classification\\5_sample"
FEATURE_DIR = "XXX\\filmgenre_classification\\7_features"
MATRIX_DIR = "XXX\\filmgenre_classification\\9_featureMatrix"
sample_metadata = "XXX\\filmgenre_classification\\5_sample\\" + SAMPLE + "_metadata.csv")
syntax_features = "XXX\\filmgenre_classification\\7_features\\" + SAMPLE + "_syn-features.csv")

metadata_df = pd.read_csv(sample_metadata)
syntax_df = pd.read_csv(syntax_features)

def create_topic_df(topics_per_document):
    topic_df = pd.read_csv(topics_per_document, delimiter="\t", header=None)
    return topic_df

def create_key_df(top_words_per_topic):
    key_df = pd.read_csv(top_words_per_topic, delimiter="\t|\s", header=None, engine='python')
    return key_df

def get_top_words_as_columns(topic_df, key_df):
	#Drop unneeded columns
    key_df.drop(key_df.columns[1], axis=1, inplace=True)
    topic_df.drop(topic_df.columns[0], axis=1, inplace=True)
    
	#Change the identifier of a subtitle file from their filepath to the IMDB-ID
    topic_df.iloc[:,0] = topic_df.iloc[:,0].str.replace("(.*?)(tt[0-9]{7}).txt", "\\2")
    
	#Insert the later needed header at the front of this list
    top_3_words_per_topic = ["IMDB-ID"]
    
    for row in key_df.itertuples():
        #Remove tuple brackets
        top_3_words_per_topic.append(", ".join([str(i) for i in row[2:5]]))
    topic_df.columns = top_3_words_per_topic
    return topic_df

def merge_dataframes(df1, df2):
    merged_matrix = pd.merge(df1, df2, on="IMDB-ID")
    return merged_matrix

def main(metadata_df, syntax_df):
    topic_df = create_topic_df(topic_distr_file)
    key_df = create_key_df(top_words_file)
    modified_topic_df = get_top_words_as_columns(topic_df, key_df)
	
	#Contains only the features but no metadata
    feature_matrix = merge_dataframes(modified_topic_df, syntax_df)

	#Master matrix contains metadata and features
    master_matrix = merge_dataframes(metadata_df, feature_matrix)
    master_matrix.to_csv(MATRIX_DIR + "\\" + SAMPLE + "_feature_matrix.csv", index=False)
main(metadata_df, syntax_df)