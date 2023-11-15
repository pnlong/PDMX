# README
# Phillip Long
# September 29, 2023

# join all the relevant information collected so far into one dataframe

# ./join_metadata.py


# IMPORTS
##################################################
import pandas as pd
##################################################


# CONSTANTS
##################################################
OUTPUT_DIR = "/data2/pnlong/musescore"
OUTPUT_FILEPATH = f"{OUTPUT_DIR}/read_mscz.csv"
##################################################


# JOINS
##################################################

df = pd.read_csv(filepath_or_buffer = f"{OUTPUT_DIR}/read_mscz_validity.csv", sep = ",", header = 0, index_col = False)

# join versions
versions = pd.read_csv(filepath_or_buffer = f"{OUTPUT_DIR}/read_mscz_versions.csv", sep = ",", header = 0, index_col = False)
df = df.merge(right = versions, how = "left", on = ("path", "is_valid"))
del versions # free up memory

# join public domain
public_domain = pd.read_csv(filepath_or_buffer = f"{OUTPUT_DIR}/public_domain.csv", sep = ",", header = 0, index_col = False)
public_domain = public_domain.rename(columns = {"data_path": "path"})
public_domain = public_domain[["path", "is_public_domain"]]
df = df.merge(right = public_domain, how = "left", on = "path")
del public_domain

# join metadata
metadata = pd.read_csv(filepath_or_buffer = f"{OUTPUT_DIR}/metadata_to_data.csv", sep = ",", header = 0, index_col = False)
metadata = metadata.rename(columns = {"data_path": "path"})
df = df.merge(right = metadata, how = "left", on = "path")
del metadata

##################################################


# OUTPUT FILE
##################################################

# reorder columns
df = df.rename(columns = {"path": "data_path"})
df = df[["data_path", "metadata_path", "data", "metadata", "is_valid", "version", "is_public_domain"]]
df.to_csv(path_or_buf = OUTPUT_FILEPATH, sep = ",", na_rep = "NA", header = True, index = False, mode = "w")

##################################################
