"""
SCRIPT 2:
This script extracts the sample metadata from the complete metadata file by comparing IMDB-IDs and saves it in the sample folder.
"""

import os
import glob
import pandas as pd 

#Change XML_DIR and SAMPLE_DIR names to the right sample folder if you have more than one sample!
SAMPLE = "sample_1930-2010"
METADATA = "XXX\\filmgenre_classification\\3_metadata\\complete_metadata.csv"
XML_DIR = "XXX\\filmgenre_classification\\5_sample\\" + SAMPLE + "\\xml"
SAMPLE_DIR = "XXX\\filmgenre_classification\\5_sample"

metadata_df = pd.read_csv(METADATA, index_col=0)
file_ids = list(metadata_df.index.values)

list_of_rows = []

for xmlfile in glob.glob(XML_DIR + "\\*.xml"):
    fullpath = os.path.basename(xmlfile)
    filename, ext = fullpath.split(".")
    if filename in file_ids:
        list_of_rows.append(metadata_df.loc[filename])
sample_metadata_df = pd.DataFrame(list_of_rows, columns=list(metadata_df))
print(sample_metadata_df)
pd.DataFrame.to_csv(sample_metadata_df, SAMPLE_DIR + "\\" + SAMPLE + "_metadata.csv", index_label="IMDB-ID")