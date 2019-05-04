"""
SCRIPT 5:
This script deletes all (empty) folders in the corpus folder.

Attention: This script does not actually check if the folders are empty! It expects that all scripts before were run successfully and in the right order.
"""

import os

ROOT_DIR = "XXX\\filmgenre_classification\\0_corpus"

counter = 0

for file_name in os.listdir(ROOT_DIR):
    
    if not os.path.isdir(os.path.join(ROOT_DIR, file_name)):
        continue
    else:
        counter = counter + 1
        print(counter, "Files removed")
        os.rmdir(os.path.join(ROOT_DIR, file_name))
        