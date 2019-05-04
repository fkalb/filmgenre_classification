"""
SCRIPT OPTIONAL:
This script creates a .csv file with information about the genre distribution over decades. The result is not needed for further scripts. It's for informational purposes only.
"""

import os
import pandas as pd

META_DIR = "XXX\\filmgenre_classification\\3_metadata"
METADATA = os.path.join(META_DIR, "complete_metadata.csv")

metadata_df = pd.read_csv(METADATA)
time_genre_df = metadata_df.loc[:, ["genres", "startYear"]]

genre_time_dict = {
                   "2010": {"Action" : 0, "Adventure" : 0, "Animation": 0, "Comedy" : 0, "Crime" : 0, "Drama" : 0, "Family": 0, "Fantasy": 0, "History" : 0, "Horror" : 0, "Musical" : 0, "Mystery" : 0, "Romance" : 0, "Sci-Fi" : 0, "Thriller" : 0 , "War": 0, "Western": 0},
                   "2000": {"Action" : 0, "Adventure" : 0, "Animation": 0, "Comedy" : 0, "Crime" : 0, "Drama" : 0, "Family": 0, "Fantasy": 0, "History" : 0, "Horror" : 0, "Musical" : 0, "Mystery" : 0, "Romance" : 0, "Sci-Fi" : 0, "Thriller" : 0 , "War": 0, "Western": 0}, 
                   "1990": {"Action" : 0, "Adventure" : 0, "Animation": 0, "Comedy" : 0, "Crime" : 0, "Drama" : 0, "Family": 0, "Fantasy": 0, "History" : 0, "Horror" : 0, "Musical" : 0, "Mystery" : 0, "Romance" : 0, "Sci-Fi" : 0, "Thriller" : 0 , "War": 0, "Western": 0}, 
                   "1980": {"Action" : 0, "Adventure" : 0, "Animation": 0, "Comedy" : 0, "Crime" : 0, "Drama" : 0, "Family": 0, "Fantasy": 0, "History" : 0, "Horror" : 0, "Musical" : 0, "Mystery" : 0, "Romance" : 0, "Sci-Fi" : 0, "Thriller" : 0 , "War": 0, "Western": 0}, 
                   "1970": {"Action" : 0, "Adventure" : 0, "Animation": 0, "Comedy" : 0, "Crime" : 0, "Drama" : 0, "Family": 0, "Fantasy": 0, "History" : 0, "Horror" : 0, "Musical" : 0, "Mystery" : 0, "Romance" : 0, "Sci-Fi" : 0, "Thriller" : 0 , "War": 0, "Western": 0},
                   "1960": {"Action" : 0, "Adventure" : 0, "Animation": 0, "Comedy" : 0, "Crime" : 0, "Drama" : 0, "Family": 0, "Fantasy": 0, "History" : 0, "Horror" : 0, "Musical" : 0, "Mystery" : 0, "Romance" : 0, "Sci-Fi" : 0, "Thriller" : 0 , "War": 0, "Western": 0},
                   "1950": {"Action" : 0, "Adventure" : 0, "Animation": 0, "Comedy" : 0, "Crime" : 0, "Drama" : 0, "Family": 0, "Fantasy": 0, "History" : 0, "Horror" : 0, "Musical" : 0, "Mystery" : 0, "Romance" : 0, "Sci-Fi" : 0, "Thriller" : 0 , "War": 0, "Western": 0}, 
                   "1940": {"Action" : 0, "Adventure" : 0, "Animation": 0, "Comedy" : 0, "Crime" : 0, "Drama" : 0, "Family": 0, "Fantasy": 0, "History" : 0, "Horror" : 0, "Musical" : 0, "Mystery" : 0, "Romance" : 0, "Sci-Fi" : 0, "Thriller" : 0 , "War": 0, "Western": 0}, 
                   "1930": {"Action" : 0, "Adventure" : 0, "Animation": 0, "Comedy" : 0, "Crime" : 0, "Drama" : 0, "Family": 0, "Fantasy": 0, "History" : 0, "Horror" : 0, "Musical" : 0, "Mystery" : 0, "Romance" : 0, "Sci-Fi" : 0, "Thriller" : 0 , "War": 0, "Western": 0}
                   }
time_dict = {
            "2010" : 0,
            "2000" : 0,
            "1990" : 0,
            "1980" : 0,
            "1970" : 0,
            "1960" : 0,
            "1950" : 0,
            "1940" : 0,
            "1930" : 0
            }

for row in time_genre_df.itertuples(index=False):
    #Get number of movies per decade
    if 2010 <= row[1] <= 2019:
            time_dict["2010"] += 1
    elif 2000 <= row[1] <= 2009:
        time_dict["2000"] += 1
    elif 1990 <= row[1] <= 1999:
        time_dict["1990"] += 1
    elif 1980 <= row[1] <= 1989:
        time_dict["1980"] += 1
    elif 1970 <= row[1] <= 1979:
        time_dict["1970"] += 1
    elif 1960 <= row[1] <= 1969:
        time_dict["1960"] += 1
    elif 1950 <= row[1] <= 1959:
        time_dict["1950"] += 1
    elif 1940 <= row[1] <= 1949:
        time_dict["1940"] += 1
    elif 1930 <= row[1] <= 1939:
        time_dict["1930"] += 1
    else:
        print("Error!")

    #Skip the header row!
    if row[0] == "genres":
        continue
    #If a row contains more than 1 Genre, split it up!
    genres = row[0].split(" ")
    for genre in genres:
        if 2010 <= row[1] <= 2019:
            genre_time_dict["2010"][genre] += 1
        elif 2000 <= row[1] <= 2009:
            genre_time_dict["2000"][genre] += 1
        elif 1990 <= row[1] <= 1999:
            genre_time_dict["1990"][genre] += 1
        elif 1980 <= row[1] <= 1989:
            genre_time_dict["1980"][genre] += 1
        elif 1970 <= row[1] <= 1979:
            genre_time_dict["1970"][genre] += 1
        elif 1960 <= row[1] <= 1969:
            genre_time_dict["1960"][genre] += 1
        elif 1950 <= row[1] <= 1959:
            genre_time_dict["1950"][genre] += 1
        elif 1940 <= row[1] <= 1949:
            genre_time_dict["1940"][genre] += 1
        elif 1930 <= row[1] <= 1939:
            genre_time_dict["1930"][genre] += 1
        else:
            print("Error!")

time_statistics_df = pd.DataFrame.from_dict(time_dict, orient="index", columns=["Filme pro Jahrzehnt"])
genre_time_statistics_df = pd.DataFrame.from_dict(genre_time_dict).T
combined_df = time_statistics_df.join(genre_time_statistics_df)
pd.DataFrame.to_csv(combined_df, META_DIR + "\\complete_genre_time_statistics.csv")