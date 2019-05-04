"""
SCRIPT 1:
This script calculates the average topic distributions and average syntactic measures.
1. Function "get_genre_averages" is used for genre averages
2. Function "get_decade_averages" is used for decade averages
3. Results are saved as .csv and .png files
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

#Set sample variable according to the FEATURE_MATRIX variable
sample = "sample_1930-2010"
MATRIX_DIR = "XXX\\filmgenre_classification\\9_featureMatrix"
RESULT_DIR = "XXX\\filmgenre_classification\\12-2_featureAverages"
FEATURE_MATRIX = MATRIX_DIR + "\\" + sample +  "_feature_matrix.csv")
matrix_df = pd.read_csv(FEATURE_MATRIX)
#pd.set_option('max_columns', 20)

plt.style.use('seaborn-whitegrid')
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_genre_averages(matrix_df):
    #genre_list = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Family", "Fantasy", "History", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    genre_list = ["Action", "Adventure", "Animation", "Crime", "Family", "Fantasy", "History", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    syns = ["ADJCL", "CNPC", "CNPS", "DPPC", "MLC", "MLNP", "MLS", "NPPS","PASSPS","SCR"]
    genre_average_topics = {}
    genre_average_doclength = {}
    genre_top_10_topics = {}

    genre_average_syn = {}

    for genre in genre_list:
        genre_df = matrix_df[matrix_df["genres"].str.contains(genre)]
        #Get topic columns and their means
        topic_columns = genre_df.iloc[:, 6:-10]
        averaged_topics = topic_columns.mean()
        genre_average_topics[genre] = averaged_topics.to_dict()

        #Get syntactic columns and their means
        syn_columns = genre_df.iloc[:, -10:]
        averaged_syn = syn_columns.mean()
        genre_average_syn[genre] = averaged_syn.to_dict()

        #Get documentLength column and the mean for each genre
        doclength_column = genre_df.iloc[:,5]
        averaged_doclength = doclength_column.mean()
        genre_average_doclength[genre] = averaged_doclength
    
    ######################
    # Topic averages     #
    ######################

    topic_average_df = pd.DataFrame.from_dict(genre_average_topics)
    #print(topic_average_df["War"].sort_values(ascending=False))
    
    for genre in genre_list:
        #Find rows with highest values for each genre
        top_10_topics = topic_average_df.nlargest(10, genre)
        #Find topic names in index of each row
        top_10_topics_list = list(top_10_topics.index.values)
        #Save top 10 topics per genre in a dict
        genre_top_10_topics[genre] = top_10_topics_list

    top10_df = pd.DataFrame.from_dict(genre_top_10_topics)
    #Change index so the best topic for each genre gets assigned a 1 instead of a 0
    top10_df.set_index(np.arange(1,11,1), inplace=True)

    pd.DataFrame.to_csv(topic_average_df, RESULT_DIR + "\\genre\\" + sample + "\\" + sample + "_topic_averages_per_genre.csv", sep="\t", index=None)
    pd.DataFrame.to_csv(top10_df, RESULT_DIR + "\\genre\\" + sample + "\\" + sample +  "_top10_topics_per_genre.csv", sep="\t")
    
    #########################
    #  Document length      #
    #########################

    #Create doclength dataframe and save it to csv
    doclength_average_df = pd.DataFrame.from_dict(genre_average_doclength, orient="index", columns=["avg_doclength"])
    pd.DataFrame.to_csv(doclength_average_df, RESULT_DIR + "\\genre\\" + sample + "\\" + sample + "_doclength_averages_per_genre.csv")
    doclength_average_df.plot(kind="bar", legend=None, figsize=(7.5,5))
    plt.xlabel("Genres", fontsize="large")
    plt.xticks(rotation=45)
    plt.ylabel("Average number of words per movie")
    plt.yticks(np.arange(0,11000,1000))
    plt.tight_layout()
    plt.savefig(RESULT_DIR + "\\genre\\" + sample + "\\" + sample "_genre_average_doclength.png")
    plt.show()


    #########################
    #  Syntactic features   #
    #########################

    #Create syntactic average dataframe and save it to csv
    syn_average_df = pd.DataFrame.from_dict(genre_average_syn)
    pd.DataFrame.to_csv(syn_average_df, RESULT_DIR + "\\genre\\" + sample + "\\" + sample + "_syntactic_averages_per_genre.csv")

    #Create ranked dataframe for easier visualization
    syn_ranked_df = syn_average_df.rank(method="max", ascending=True, axis=1)
    
    #Insert syntactic feature labels into the dataframe again, because they can't be inferred from the index itself using the parallel_coordinates function!
    syn_ranked_df["syns"] = syns

    #Create parallel_coordinates figure
    plt.figure(figsize=(7.5,5))
    pd.plotting.parallel_coordinates(syn_ranked_df, "syns", colormap="Dark2")
    plt.ylabel("Inverted rank", fontsize="large")
    plt.yticks(np.arange(1,16,2))
    plt.xlabel("Genres", fontsize="large")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(RESULT_DIR + "\\genre\\" + sample + "\\" + sample + "_genre_syn_parallel_plot.png", bbox_inches="tight")
    plt.show()

    #Create pandas series with average ranks per genre
    syn_avg_ranks = syn_ranked_df.mean()

    #Create barplot with average ranks per genre
    plt.figure(figsize=(7.5,5))
    syn_avg_ranks.plot(kind="bar", legend=None)
    plt.ylabel("Average inverted rank", fontsize="large")
    plt.xlabel("Genres", fontsize="large")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULT_DIR  + "\\genre\\" + sample + "\\" + sample + "_genre_syn_average_ranks.png", bbox_inches="tight")
    plt.show()

def get_decade_averages(matrix_df):
    
    #Get the right list of decades depending on the used sample
    if sample == "sample_1930-2010" or sample == "sample_genre-based":
        decade_list = ["1930", "1940", "1950", "1960", "1970", "1980", "1990", "2000", "2010"]
    elif sample == "sample_1970-2010":
        decade_list = ["1970", "1980", "1990", "2000", "2010"]
    elif sample == "sample_2000-2010":
        decade_list = ["2000", "2010"]
    
    syns = ["ADJCL", "CNPC", "CNPS", "DPPC", "MLC", "MLNP", "MLS", "NPPS","PASSPS","SCR"]

    decade_average_topics = {}
    decade_average_doclength = {}
    decade_top_10_topics = {}

    decade_average_syn = {}

    for decade in decade_list:
        #Cast on "int" is necessary because otherwise wrong types (int vs string) are compared
        decade_df = matrix_df.loc[matrix_df["decade"] == int(decade)]
        #Get topic columns and their means
        topic_columns = decade_df.iloc[:, 6:-10]
        averaged_topics = topic_columns.mean()
        decade_average_topics[decade] = averaged_topics.to_dict()

        #Get syntactic columns and their means
        syn_columns = decade_df.iloc[:, -10:]
        averaged_syn = syn_columns.mean()
        decade_average_syn[decade] = averaged_syn.to_dict()

        #Get documentLength column and the mean for each genre
        doclength_column = decade_df.iloc[:,5]
        averaged_doclength = doclength_column.mean()
        decade_average_doclength[decade] = averaged_doclength
    
    ######################
    # Topic averages     #
    ######################

    topic_average_df = pd.DataFrame.from_dict(decade_average_topics)

    for decade in decade_list:
        #Find rows with highest values for each genre
        top_10_topics = topic_average_df.nlargest(10, decade)
        #Find topic names in index of each row
        top_10_topics_list = list(top_10_topics.index.values)
        #Save top 10 topics per genre in a dict
        decade_top_10_topics[decade] = top_10_topics_list

    top10_df = pd.DataFrame.from_dict(decade_top_10_topics)
    #Change index so the best topic for each genre gets assigned a 1 instead of a 0
    top10_df.set_index(np.arange(1,11,1), inplace=True)
    
    pd.DataFrame.to_csv(topic_average_df, RESULT_DIR + "\\decade\\" + sample + "\\" + sample + "_topic_averages_per_decade.csv", sep="\t", index=None)
    pd.DataFrame.to_csv(top10_df, RESULT_DIR + "\\decade\\" + sample + "\\" + sample + "_top10_topics_per_decade.csv", sep="\t")

    #########################
    #  Document length      #
    #########################

    #Create doclength dataframe and save it to csv
    doclength_average_df = pd.DataFrame.from_dict(decade_average_doclength, orient="index", columns=["avg_doclength"])
    pd.DataFrame.to_csv(doclength_average_df, RESULT_DIR + "\\decade\\" + sample + "\\" + sample + "_doclength_averages_per_decade.csv")
    doclength_average_df.plot(kind="bar", legend=None, figsize=(7.5,5))
    plt.xlabel("Decades", fontsize="large")
    plt.xticks(rotation=45)
    plt.ylabel("Average number of words per movie")
    plt.yticks(np.arange(0,11000,1000))
    plt.tight_layout()
    plt.savefig(RESULT_DIR + "\\decade\\" + sample + "\\" + sample + "_decade_average_doclength.png")
    plt.show()

    #########################
    #  Syntactic features   #
    #########################

    #Create syntactic average dataframe and save it to csv
    syn_average_df = pd.DataFrame.from_dict(decade_average_syn)
    pd.DataFrame.to_csv(syn_average_df, RESULT_DIR + "\\decade\\" + sample + "\\" + sample + "_syntactic_averages_per_decade.csv")

    #Create ranked dataframe for easier visualization
    syn_ranked_df = syn_average_df.rank(method="max", ascending=True, axis=1)
    
    #Insert syntactic feature labels into the dataframe again, because they can't be inferred from the index itself using the parallel_coordinates function!
    syn_ranked_df["syns"] = syns

    #Create parallel_coordinates figure
    plt.figure(figsize=(7.5,5))
    pd.plotting.parallel_coordinates(syn_ranked_df, "syns", colormap="Dark2")
    plt.ylabel("Inverted rank", fontsize="large")
    plt.yticks(np.arange(1,len(decade_list),1))
    plt.xlabel("Decades", fontsize="large")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(RESULT_DIR + "\\decade\\" + sample + "\\" + sample + "_decade_syn_parallel_plot.png", bbox_inches="tight")
    plt.show()

    #Create pandas series with average ranks per genre
    syn_avg_ranks = syn_ranked_df.mean()

    #Create barplot with average ranks per genre
    plt.figure(figsize=(7.5,5))
    syn_avg_ranks.plot(kind="bar", legend=None)
    plt.ylabel("Average inverted rank", fontsize="large")
    plt.xlabel("Decades", fontsize="large")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULT_DIR  + "\\decade\\" + sample + "\\" + sample + "_decade_syn_average_ranks.png", bbox_inches="tight")
    plt.show()

def main(matrix_df):
    get_genre_averages(matrix_df)
    get_decade_averages(matrix_df)

main(matrix_df)