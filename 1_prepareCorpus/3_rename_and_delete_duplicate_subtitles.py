"""
SCRIPT 3:
This script searches for the largest subtitle file of each film and renames it to match the corresponding IMDB-ID. The duplicates are deleted afterwards.

NOTE: The NEXT script (4_move_files_one_level_up.bat) has to be executed from the /0_corpus folder and is also located there.
"""

import os

ROOT_DIR = "XXX\\filmgenre_classification\\0_corpus"

def find_largest_file_and_rename(ROOT_DIR):
    counter = 0
    for dir_path, dir_names, file_names in os.walk(ROOT_DIR):
        #Skip top directory
        if dir_path == ROOT_DIR:
            continue
        print("Current directory:", dir_path)
        file_dict = {}
		#Saves the filesize of every subtitle archive in a dictionary
        for file in file_names:
            stats = os.stat(os.path.join(dir_path, file))
            file_size = stats.st_size
            file_dict[file] = file_size
		#Look for the largest file in the dict
        largest_file = max(file_dict.items(), key= lambda x: x[1])[0]
		#Rename the file to match the IMDB_ID instead of the OpenSubtitles-ID
        os.rename(os.path.join(dir_path, largest_file), os.path.join(dir_path, os.path.basename(os.path.normpath(dir_path)) + ".xml.gz"))

def delete_files_except_largest(ROOT_DIR):
    for dir_path, dir_names, file_names in os.walk(ROOT_DIR):
        #Skip top directory
        if dir_path == ROOT_DIR:
            continue
        print("Current directory:", dir_path)
		#If folder name and subtitle archive name are equal (=IMDB-ID) skip them. Duplicates get deleted
        for file in file_names:
            if file.split(".")[0] == os.path.basename(os.path.normpath(dir_path)):
                print("Datei und Ordner gleich!")
            else:
                os.remove(os.path.join(dir_path, file))

def main(ROOT_DIR):
    find_largest_file_and_rename(ROOT_DIR)
    print("Largest files found and renamed. Now starting to delete the rest!")
    delete_files_except_largest(ROOT_DIR)
    print("All duplicates deleted!")
main(ROOT_DIR)