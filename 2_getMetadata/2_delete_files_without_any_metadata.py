"""
SCRIPT 2:
This script looks for movies where no metadata is available and deletes them afterwards.

NOTE: Be careful before running scripts that delete files! Always check the IDs of the files (comment the remove command in line 34) that you are about to delete and keep backups (speaking from experience)
"""

import os
import glob
import pandas as pd 

META_DIR = "XXX\\filmgenre_classification\\2_metadata"
FILE_DIR = "XXX\\filmgenre_classification\\0_corpus"
METADATA = os.path.join(META_DIR, "complete_metadata.csv")
METADATA_DF = pd.read_csv(METADATA)

list_with_good_ids = METADATA_DF["IMDB-ID"].tolist()
list_with_all_ids = []

for xmlfile in glob.glob(FILE_DIR + "\*.xml"):
    base = os.path.basename(xmlfile)
    filename, extension = base.split(".")
    list_with_all_ids.append(filename)

difference_between_lists = [x for x in list_with_all_ids if x not in list_with_good_ids]
print(difference_between_lists)

counter = 0
for xmlfile in glob.glob(FILE_DIR + "\*.xml"):
    base = os.path.basename(xmlfile)
    filename, extension = base.split(".")
    if filename in difference_between_lists:
        counter = counter + 1
        os.remove(os.path.join(FILE_DIR, base))
        #print(os.path.join(FILE_DIR, base))
        print(counter, "Files removed")
