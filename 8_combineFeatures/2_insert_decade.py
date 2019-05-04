"""
SCRIPT 2:
This script inserts the decade of the film into the feature matrix.
Inserting the decade is needed for the decade-based classification.
"""

import os
import glob
import pandas as pd 
pd.set_option('display.max_columns', 12)

sample = "sample_1930-2010"
MATRIX_FILE = "XXX\\filmgenre_classification\\9_featureMatrix\\" + SAMPLE + "_feature_matrix.csv"

matrix_df = pd.read_csv(MATRIX_FILE)
years = list(matrix_df["startYear"])
decades = [(year - (year % 10)) for year in years]
decade_series = pd.Series(decades)
#Insert decade as the 4th column, so that features start in column 6
matrix_df.insert(4, "decade", decade_series)
#Save matrix to file
pd.DataFrame.to_csv(matrix_df, MATRIX_FILE, index=False)
print("Decade inserted!")