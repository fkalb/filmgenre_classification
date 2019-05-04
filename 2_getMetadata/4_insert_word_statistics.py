"""
SCRIPT 5:
This script inserts the document length (excluding punctuation and other symbols) of each subtitle file into the metadata file.
"""

import os
import glob
import pandas as pd
import lxml.etree

FILE_DIR = "XXX\\filmgenre_classification\\0_corpus"
META_DIR = "XXX\\filmgenre_classification\\2_metadata"
METADATA = os.path.join(META_DIR, "complete_metadata.csv")

metadata_df = pd.read_csv(METADATA)
word_counts = {}

for xml_file in glob.glob(FILE_DIR + r"\*.xml"):
    base = os.path.basename(xml_file)
    filename, extension = base.split(".")
    print("Processing file", filename)
    
    xmltree = lxml.etree.parse(xml_file)
    number_of_w_elements = xmltree.xpath("count (//w)")
    number_of_punctuation = xmltree.xpath("count (//w[@xpos='PUNCT'])")
    #Symbols like a musical note are also subtracted
    number_of_symbols = xmltree.xpath("count (//w[@xpos='SYM'])")
    word_counts[filename] = int(number_of_w_elements) - int(number_of_punctuation) - int(number_of_symbols)

wordcount_frame = pd.DataFrame.from_dict(word_counts, orient="index", columns=["documentLength"])
merged_frame = metadata_df.join(wordcount_frame, how="inner", on="IMDB-ID")
#print(merged_frame)
merged_frame.to_csv(METADATA, index=False)
