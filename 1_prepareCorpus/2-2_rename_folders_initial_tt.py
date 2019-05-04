"""
SCRIPT 2.2:
This script inserts the string "tt" at the beginning. Folder names now represent valid IMDB-IDs.
"""

import os

ROOT_DIR = "XXX\\filmgenre_classification\\0_corpus"

counter = 0

for file_name in os.listdir(ROOT_DIR):
    if not os.path.isdir(os.path.join(ROOT_DIR, file_name)):
        continue
    if 'tt' in file_name:
        continue
    if 'tt' not in file_name:    
        os.rename(os.path.join(ROOT_DIR, file_name), os.path.join(ROOT_DIR, "tt" + file_name))
        counter = counter + 1
        print(counter, "Files successfully renamed!")