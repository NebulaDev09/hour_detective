import pandas as pd

data = pd.read_csv("train.csv")

# finding empty cells
missing_values = data.isna().sum()
print(missing_values[missing_values > 0])

#some cells like load_time_ms can be empty which means the website didnt respond 
# but cells like churn_ratio shouldnt be empty ideally, so I will decide some core columns which when empty,
# I will disregard the whole row

"""churn_ratio                   21
pct_code_in_first_commit     186
pct_primary_language         194
avg_file_length              194
comment_line_ratio           194
has_custom_art                 2
page_weight_kb              1349
num_js_files_loaded         1423
num_internal_pages          1423
is_mobile_responsive        1423
load_time_ms                1349
project_category               5
is_multiplayer                 5
mentions_ai_api                3
is_clone_or_tutorial           5""" # output of missing values

core_cols = [
    "churn_ratio",
    "pct_code_in_first_commit",
    "avg_file_length",
    "comment_line_ratio",
] # some cols important to the fraud detecton that ideally shouldnt be empty

missing_core_values = data[core_cols].isna().any(axis=1) # this gave 195
print(missing_core_values.sum())

data_cleaned = data.dropna(subset=core_cols) # this drops our whole data to 4305 rows
print(len(data_cleaned))