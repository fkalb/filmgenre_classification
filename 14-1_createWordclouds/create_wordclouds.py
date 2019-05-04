"""
SCRIPT 1:
This script takes the wordweights-file for each sample and creates word clouds. The wordweight-files that resulted from my Topic Modeling experiments are too large for
uploading them on GitHub so the word clouds themselves are uploaded. If you want to create new word clouds, you need to run a Topic Modeling process with Mallet first.

Attention: The overwhelming logical part of this script was borrowed from the following script written by Christof Schöch: https://github.com/cligs/projects/blob/master/2015/gddh/code/visualize.py
and then adapted to the data at hand. Some adjustments had to be made, e.g.: the read_table function by pandas is no longer the preferred option for reading input data, 
so read_csv was used, which lead to various changes in the code.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import seaborn as sns


SAMPLES = ["1930-2010", "1970-2010", "2000-2010", "genre-based"]
CLOUD_DIR = "XXX\filmgenre_classification\14-2_wordclouds"
WORDWEIGHTS_DIR = "XXX\filmgenre_classification\7_features"
WORDS = 40
COLUMN_NAMES = ["topic", "word", "score"]
#The font file was not included into the repository, because the legal status for sharing it, is not clear. Just make sure your font-file matches the file path
FONT_PATH = WORDWEIGHTS_DIR_DIR + "\\fonts\AppleGaramond.ttf"

def read_sample_file(sample, word_weights_file, WORDS, COLUMN_NAMES):
    wordweights_df = pd.read_csv(word_weights_file, names=COLUMN_NAMES, usecols=COLUMN_NAMES, sep="\t")
    #Check if Dataframe looks correct
    #print(wordweights_df.head())

    #Extract number of topics for each sample
    number_of_topics = len(set(wordweights_df.iloc[:,0]))

    #Group dataframe by topics
    wordweights_df_grouped = wordweights_df.sort_values("score", ascending=False).groupby("topic")

    for topic in range(0,number_of_topics):
        words_for_wordcloud, top_3_words = get_topic_data(wordweights_df_grouped, WORDS, topic)
        create_wordcloud(sample, topic, words_for_wordcloud, top_3_words, number_of_topics)

def get_topic_data(grouped_word_weights_df, WORDS, topic):

    words_for_wordcloud = ""
    #This contains the words and scores for each topic. Scores are descending
    words_and_scores = grouped_word_weights_df.get_group(topic)
    #Get only the top 40 words and their scores
    top_words_and_scores = words_and_scores.iloc[0:WORDS]
    #Save the top words
    top_words = top_words_and_scores.iloc[:,1].tolist()
    #Get the top 3 words for naming the files
    top_3_words = "-".join(top_words[0:3])
    #Save the top scores
    top_scores = top_words_and_scores.iloc[:,2].tolist()
    #Put top words in a string multiplied by their score to account for their importance.
    #This string is basically a text which is then used for creating word clouds, which are based on word frequencies
    i = 0
    for word in top_words:
        score = top_scores[i]
        words_for_wordcloud = words_for_wordcloud + ((word + " ") * int(score))
        i += 1
    return words_for_wordcloud, top_3_words

def create_wordcloud(sample, topic, words_for_wordcloud, top_3_words, number_of_topics):
    
    wordcloud = WordCloud(font_path=FONT_PATH, width=1200, height=1000, background_color="white", color_func=lambda *args, **kwargs: "#2c6fbb", collocations=False, stopwords=None).generate(words_for_wordcloud)
    topic_number = topic + 1
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Sample: " + sample + " Topic: " + str(topic) + " [" +  str(topic_number) + "/" + str(number_of_topics) + "]" + "\n", fontsize=20)
    plt.tight_layout()
    plt.savefig(CLOUD_DIR + "\\" + sample + "\\" + str(topic) + "_" + top_3_words + ".png")
    plt.show()

def main(WORDS, COLUMN_NAMES):
    for word_weights_file, sample in zip(glob.glob(WORDWEIGHTS_DIR + "\*_wordweights.csv"), SAMPLES):
        if not os.path.exists(CLOUD_DIR + "\\" + sample):
            os.makedirs(CLOUD_DIR + "\\" + sample)
        read_sample_file(sample, word_weights_file, WORDS, COLUMN_NAMES)
main(WORDS, COLUMN_NAMES)
