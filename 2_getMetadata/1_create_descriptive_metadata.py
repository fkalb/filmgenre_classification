"""
SCRIPT 1:
This script creates a metadata file in .csv format for all films in the corpus. 
To get the metadata for the movies from the IMDB:
1. Download the following file (title.basics.tsv.gz) from "https://datasets.imdbws.com/" 
2. Unzip into the /2_metadata folder and rename it to imdb_metadata.tsv
"""

import os
import csv
import glob
import pandas as pd 

META_DIR = "XXX\\filmgenre_classification\\2_metadata"
FILE_DIR = "XXX\\filmgenre_classification\\0_corpus"
IMDB_IDS_FROM_DATASET = os.path.join(META_DIR, "imdb_ids.csv")
IMDB_METADATA = os.path.join(META_DIR, "imdb_metadata.tsv")

def create_file():
    with open(IMDB_IDS_FROM_DATASET, "w", encoding="UTF-8") as csv_file:
        print("File created!")

def create_header():
    with open(IMDB_IDS_FROM_DATASET, "w", encoding="UTF-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter="\t")
        writer.writerow(["IMDB-ID"])
    print("Header created!")

def get_ids_from_filenames(FILE_DIR):
    for xml_file in glob.glob(FILE_DIR + "\*.xml"):
        base = os.path.basename(xml_file)
        imdb_id, extension = base.split(".")
        with open(IMDB_IDS_FROM_DATASET, "a", encoding="UTF-8", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter="\t")
            writer.writerow([imdb_id])

def merge_dataframes(IMDB_IDS_FROM_DATASET, IMDB_METADATA):
    df1 = pd.read_csv(IMDB_IDS_FROM_DATASET)
    df2 = pd.read_csv(IMDB_METADATA, sep="\t")
    df_merge = pd.merge(df1,df2,on="IMDB-ID")
    #print(df_merge.head)
    df_only_wanted_columns = df_merge.drop(["titleType", "primaryTitle", "isAdult", "endYear", "runtimeMinutes"], axis=1)
    df_only_wanted_columns.to_csv(META_DIR + "\complete_metadata.csv", index=False)

def main(META_DIR, FILE_DIR, IMDB_IDS_FROM_DATASET, IMDB_METADATA):
    if not os.path.exists(IMDB_IDS_FROM_DATASET):
        create_file()
    if os.stat(IMDB_IDS_FROM_DATASET).st_size == 0:
        create_header()
    get_ids_from_filenames(FILE_DIR)
    merge_dataframes(IMDB_IDS_FROM_DATASET,IMDB_METADATA)

main(META_DIR, FILE_DIR, IMDB_IDS_FROM_DATASET, IMDB_METADATA)
    