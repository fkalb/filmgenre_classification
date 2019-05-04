"""
SCRIPT 3:
This script extracts Nouns, Adjectives, Adverbs and Verbs from the subtitle .xml files and saves them in .txt files.
The .txt files are then used by Mallet for topic modeling.

NOTE: ALWAYS RUN FROM TERMINAL (Executing from VisualStudioCode did not always work as expected)
"""

import os
import lxml.etree
import glob

SAMPLE = "sample_1930-2010"
XML_DIR = "XXX\\filmgenre_classification\\5_sample\\" + SAMPLE + "\\xml"
TXT_DIR = "XXX\\filmgenre_classification\\5_sample\\" + SAMPLE + "\\txt"

if not os.path.exists(TXT_DIR):
    os.makedirs(TXT_DIR)

for xml_file in glob.glob(XML_DIR + "\\*.xml"):
    
    filebase = os.path.basename(xml_file)
    filename, extension = filebase.split(".")
    print("Now processing:", filename)
    xml_tree = lxml.etree.parse(xml_file)
    xml_root = xml_tree.getroot()
    text = []
    for word in xml_root.iter("w"):
        #text.append(word)
        #Proper nouns like names don't work well for topic modelling! So they are not included. There is also a check if the word only consists of alphabetic characters, because sometimes # are found as words
        if (word.get("xpos") == "NOUN" or word.get("xpos") == "ADJ" or word.get("xpos") == "ADV" or word.get("xpos") == "VERB") and word.text.isalpha():
            text.append(word.text.lower())
        else: 
            continue

    text_as_str = " ".join(text)
    with open(TXT_DIR + "\\" + filename + ".txt", "w", encoding="utf-8") as txt_file:
        txt_file.write(text_as_str)
    