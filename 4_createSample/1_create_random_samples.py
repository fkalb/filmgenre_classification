"""
SCRIPT 1:
This script is used for creating random samples from the complete corpus. One of the two functions must be chosen:
1. "create_genre_based_sample": With this function genres can be excluded when creating a sample. In my case Drama and Comedy
(the two largest genres) have not been included in one sample for balance reasons. You can add other genres buy adding more if-statements or changing the present ones!
2. "create_decade_based_sample": With this function certain decades can be excluded. After that an even number of movies per decade will be extracted.
NOTE: Movie distribution over decades is highly unbalanced towards newer movies! 

Overview of movies per decade:
	1930:	971
	1940:	1015
	1950:	1691
	1960:	2788
	1970:	3250
	1980:	3403
	1990:	4551
	2000:	11219
	2010:	11743

	When choosing "amount_of_movies" make sure there are enough movies per decade.
"""

import os
from itertools import chain
import glob
import pandas as pd
import random
import shutil

#Name your sample as it will create a special folder for it!
SAMPLE = "sample_1930-2010"
FILE_DIR = "XXX\\filmgenre_classification\\0_corpus"
META_DIR = "XXX\\filmgenre_classification\\3_metadata"
SAMPLE_DIR = "XXX\\filmgenre_classification\\4_sample\\" + SAMPLE + "\\xml"
METADATA = os.path.join(META_DIR, "complete_metadata.csv")

if not os.path.exists(SAMPLE_DIR):
    os.makedirs(SAMPLE_DIR)

#################   PARAMETER   ################
meta_dataframe = pd.read_csv(METADATA)
#All years contains years from 1988 to 2017
decade_list = [1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
#This is calculated as "decades_wanted = decade_list - unwanted_decades" due to the nature of sampling. ex: For using 9 decades substract nothing!
#As an alternative approach the decade list can be changed if a chronological list (ex. 1970-2010) is not wanted
unwanted_decades = 0

decades = decade_list[unwanted_decades : 9]


#Check overview on top of the file (or genre_time_statistics.csv if you created that) first to see if you have enough files per decade
amount_of_movies = 500
#############################################

#ALWAYS START FILE FROM TERMINAL OR LESS FILES WILL BE COPIED (This is true if you use VisualStudioCode like me)

def create_genre_based_sample(meta_dataframe):
    major_genre_movies = []

    id_genre_df = meta_dataframe.loc[:, ["IMDB-ID", "genres"]]
    #print(id_genre_df)
    counter = 0
    for row in id_genre_df.itertuples(index=False):
        #Skip the header row
        if row[1] == "genres":
            continue
        genres = row[1].split(" ")
        #print(genres)
        if "Drama" in genres:
            major_genre_movies.append(row[0])
            continue
        if "Comedy" in genres:
            major_genre_movies.append(row[0])
            continue
    all_ids = list(id_genre_df["IMDB-ID"])
    #Get all movies that are not Drama or Comedy
    minor_genre_movies = [x for x in all_ids if x not in major_genre_movies]
    #print(len(minor_genre_movies), "files to be copied")
    return minor_genre_movies

def create_decade_based_sample(meta_dataframe, amount_of_movies, decades):
    random_movies = {}
    for decade in decades:
        ids_per_decade = []
        #End of range is not inclusive, so it needs to be 10 for getting all years in a decade.
        for year in range(decade, decade + 10):
            movies_per_decade = meta_dataframe[(meta_dataframe["startYear"] == year)]
            ids_per_year = list(movies_per_decade["IMDB-ID"])
            ids_per_decade.append(ids_per_year)
        #Flatten list of lists to list
        ids_per_decade = list(chain.from_iterable(ids_per_decade))
        random_movies[decade] = random.sample(ids_per_decade, amount_of_movies)

    movies_to_copy = []
    for decade, samples_per_decade in random_movies.items():
        for movie in samples_per_decade:
            movies_to_copy.append(movie)
    print(len(movies_to_copy))
    return movies_to_copy

def copy_files(sampled_movies, destination):
    counter = 0
    for xmlfile in glob.glob(FILE_DIR + r"\*.xml"):
        fullpath = os.path.basename(xmlfile)
        filename, extension = fullpath.split(".")
        if filename in sampled_movies:
            shutil.copy2(xmlfile, destination)
            counter += 1
            print(counter, "Files copied!")

def main(FILE_DIR, META_DIR, SAMPLE_DIR, meta_dataframe, amount_of_movies, decades):
    sampled_movies = create_decade_based_sample(meta_dataframe, amount_of_movies, decades)
    sampled_movies = create_genre_based_sample(meta_dataframe)
    copy_files(sampled_movies, SAMPLE_DIR)
main(FILE_DIR, META_DIR, SAMPLE_DIR, meta_dataframe, amount_of_movies, decades)