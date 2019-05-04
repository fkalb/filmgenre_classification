"""
Optional:
This script visualizes genre and decade based statistics and saves them as .png files.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt 

plt.style.use("seaborn-whitegrid")

META_DIR = "XXX\filmgenre_classification\3_metadata"
GENRE_DIR = "XXX\filmgenre_classification\3_metadata\genre_based"
DECADE_DIR = "XXX\filmgenre_classification\3_metadata\decade_based"
GENRE_TIME_FILE = os.path.join(META_DIR, "complete_genre_time_statistics.csv")

genre_time_df = pd.read_csv(GENRE_TIME_FILE, index_col=0)

data_per_genre = genre_time_df.iloc[:,1:]
amount_of_movies_per_decade = genre_time_df.iloc[:, 0]
decade_iterator = 0

###########
# OVERALL #
###########
amount_of_movies_per_decade.plot.line()
plt.xlabel("Decades", fontsize="large")
plt.ylabel("Number of movies per decade", fontsize="large")
plt.title("Overall development")
plt.tight_layout()
plt.savefig(META_DIR + "\\overall_development_of_movie_numbers.png")
plt.show()

##################
# DECADE LEVEL   #
##################
for label, content in data_per_genre.iterrows():
    content.plot.bar()
    plt.title("Decade: " + str(label) + "-" + str(label + 9), fontsize="large", loc="left")
    plt.title("Unique movies: " + str(amount_of_movies_per_decade.iloc[decade_iterator]), loc="right", fontsize="medium")
    plt.xlabel("Genres", fontsize="large")
    plt.ylabel("Number of movies per genre", fontsize="large")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(META_DIR + "\\number_of_movies_per_genre_in_" + str(label) + ".png")
    plt.show()
    decade_iterator += 1

##################
# GENRE LEVEL    #
##################

for label,content in data_per_genre.iteritems():
   
    content.plot.line()
    plt.title(label)
    plt.xlabel("Decades", fontsize="large")
    plt.xticks(rotation=45)
    plt.ylabel("Number of movies per decade", fontsize="large")
    plt.tight_layout()
    plt.savefig(META_DIR + "\\number_of_movies_per_decade_" + str.upper(label) + ".png")
    plt.show()
    decade_iterator += 1
