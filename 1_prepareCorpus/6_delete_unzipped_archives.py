"""
SCRIPT 6:
This script deletes all archives that are not needed anymore in the /0_corpus folder. 

Attention: Make sure every archive is unzipped and ready to be deleted before you execute this script!
"""

import os
import glob

ROOT_DIR = "XXX\\filmgenre_classification\\0_corpus"

counter = 0

for archive in glob.glob(ROOT_DIR + "\*.gz"):
    print("Deleting:", archive)
    os.remove(os.path.join(ROOT_DIR, archive))