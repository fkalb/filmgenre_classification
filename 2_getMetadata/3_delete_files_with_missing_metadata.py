"""
SCRIPT 3:
This script's main task is to search for movies with missing metadata and delete them (Comment out the methods you don't need in the main-method)

Another task was to recognize TV-Series with the "title.episode.tsv.gz" file from "https://datasets.imdbws.com/" (file has to be named "imdb_episode_metadata.tsv" and put into the 3_metadata folder after downloading)
During this task a new metadata file is created with a note ("DELETE"). 
While the files can be deleted though running this script, the entries in the metadata file have to be deleted manually in the .csv file.

The same procedure was used to delete movies with genres, release years or a document length (word count) that were not wanted.

Example: If all movies before 1930 should be deleted, the startYear cells in the metadata file were emptied manually so they count as files with missing metadata. After that
the script could be used to delete those files automatically. In those cases the cells are checked for being NULL. If metadata in the original IMDB-File was missing, it was noted with "\N".
This is NOT a good solution (but an easy one) and will eventually be improved/changed in the future.

NOTE: Be careful before running scripts that delete files! Always double check the IDs of the files that you are about to delete and keep backups (speaking from experience)! 
"""

import os
import glob
import pandas as pd 

META_DIR = "XXX\\filmgenre_classification\\2_metadata"
FILE_DIR = "XXX\\filmgenre_classification\\0_corpus"
METADATA = os.path.join(META_DIR, "complete_metadata.csv")
EPISODE_METADATA = os.path.join(META_DIR, "imdb_episode_metadata.tsv")


def search_tv_series(METADATA, EPISODE_METADATA):
    metadata_df = pd.read_csv(METADATA)
    episode_df = pd.read_csv(EPISODE_METADATA, sep="\t", low_memory=False)

    dataset_ids = list(metadata_df["IMDB-ID"])
    episode_ids = list(episode_df["tconst"])
    series_ids = list(episode_df["parentTconst"])
    
    ids_to_delete = list(set(episode_ids).intersection(dataset_ids))
    print(len(ids_to_delete), "common IDs")
    
    for ID in ids_to_delete:
        metadata_df.loc[(metadata_df["IMDB-ID"] == ID), "originalTitle"] = "DELETE"
    pd.DataFrame.to_csv(metadata_df, META_DIR + "\\metadata_changed.csv")
    return ids_to_delete

def search_ids_without_genre(METADATA):
    metadata_df = pd.read_csv(METADATA)
    df_without_genres = metadata_df[(metadata_df["genres"] == r"\N")] #\N am Anfang
    #df_without_genres = metadata_df[(metadata_df["genres"].isnull())]
    ids_to_delete = df_without_genres["IMDB-ID"]
    print(len(ids_to_delete))
    return list(ids_to_delete)

def search_ids_without_title(METADATA):
    metadata_df = pd.read_csv(METADATA)
	df_without_titles = metadata_df[(metadata_df["originalTitle"] == r"\N"]
    #df_without_titles = metadata_df[(metadata_df["originalTitle"].isnull())]
    ids_to_delete = df_without_titles["IMDB-ID"]
    print(ids_to_delete)
    return list(ids_to_delete)

def search_ids_without_release(METADATA):
    metadata_df = pd.read_csv(METADATA)
    df_without_release = metadata_df[(metadata_df["startYear"] == r"\N")]
    #df_without_release = metadata_df[(metadata_df["startYear"].isnull())]
    ids_to_delete = df_without_release["IMDB-ID"]
    print(len(ids_to_delete))
    return list(ids_to_delete)

def search_ids_without_wordcount(METADATA):
    metadata_df = pd.read_csv(METADATA)
    df_without_wordcount = metadata_df[(metadata_df["words"].isnull())]
    ids_to_delete = df_without_wordcount["IMDB-ID"]
    print(ids_to_delete)
    return list(ids_to_delete)

def delete_files(ids_to_delete):
    counter = 0
    for xml_file in glob.glob(FILE_DIR + "\*.xml"):
        base = os.path.basename(xml_file)
        filename, extension = base.split(".")
        if filename in ids_to_delete:
            counter = counter + 1
            os.remove(os.path.join(FILE_DIR, base))
            print(os.path.join(FILE_DIR, base))
            print(counter, "Files removed")

def main(FILE_DIR, META_DIR, METADATA, EPISODE_METADATA):
    ids_to_delete_series = search_tv_series(METADATA, EPISODE_METADATA)
	delete_files(ids_to_delete_series)
    ids_to_delete_title = search_ids_without_title(METADATA)
    delete_files(ids_to_delete_title)
    ids_to_delete_genre = search_ids_without_genre(METADATA)
    delete_files(ids_to_delete_genre)
    ids_to_delete_release = search_ids_without_release(METADATA)
    delete_files(ids_to_delete_release)
    ids_to_delete_wordcount = search_ids_without_wordcount(METADATA)
    delete_files(ids_to_delete_wordcount)
main(FILE_DIR,META_DIR,METADATA, EPISODE_METADATA)