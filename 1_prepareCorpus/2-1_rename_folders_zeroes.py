"""
SCRIPT 2.1:
This script renames the folder files by inserting zeroes at the beginning, because correct IMDB-IDs have a length of 7 characters.
"""

import os

ROOT_DIR = "XXX\\filmgenre_classification\\0_corpus"

id_length = 7
counter = 0

for file_name in os.listdir(ROOT_DIR):
    if not os.path.isdir(os.path.join(ROOT_DIR, file_name)):
        continue
    if len(file_name) == id_length:
        continue
    difference = id_length - len(file_name)
    amount_of_zeroes = difference * str(0)
    os.rename(os.path.join(ROOT_DIR, file_name), os.path.join(ROOT_DIR, amount_of_zeroes + file_name))
    counter = counter + 1
    print(counter, "File successfully renamed!")
