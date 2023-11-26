from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from urllib.parse import urlencode
from copy import deepcopy as dc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import requests
import json
import re

# data import set up
prefix = "./data/raw/"
csvs = [
    "CPI.csv",
    "1980 - 2020 NYC State and Federal Budget.csv",
    "2006 - 2012 ELA (All Students).csv",
    "2006 - 2012 ELA (Ethnicity).csv",
    "2006 - 2012 ELA (SWD).csv",
    "2013 - 2018 ELA.csv",
    "2019 ELA.csv",
    "2021 ELA.csv",
    "NYC 2021 ELA.csv"
]



"""CPI Cleaning

    Raw Data Desc
    ---------------
    Year: the year (2006-2021)
    Jan-Dec: a column for the CPI for each month
    Annual: annual CPI
    HALF1: CPI for months 1-6
    HALF2: CPI for months 7-12
    
    Clean Data Desc
    ---------------
    Indexed by Year (2006-2020)
    Annual: annual CPI
"""
cpi = pd.read_csv(prefix + csvs[0])
cpi = cpi.set_index("Year", drop = True)
# drop 21 since budget goes to 20
cpi = cpi.drop(labels = 2021)
# keep only Annual
cpi = cpi["Annual"].to_frame()


"""Budget Cleaning

    Raw Data Desc
    ---------------
    State and Federal Categorical Aid: the category where the budget is allocated
    source_of_categorical_aid: "State" or "Federal"
    FY XXXX: Fiscal Year where XXXX represents one of 1980-2020
    
    Clean Data Desc
    ---------------
    Indexed by Year (2006-2020)
    State: state allocated education budget
    Federal: federal allocated education budget
"""
budget = pd.read_csv(prefix + csvs[1])
# only get education
budget = budget[budget["State and Federal Categorical Aid"] == "Education"]

# transpose and set header
budget = budget.transpose()
budget = budget.rename(columns=budget.iloc[1])

# drop category rows
budget = budget.drop(labels=["State and Federal Categorical Aid", "source_of_categorical_aid"])

# rename FY index
keys = ";".join(budget.index.tolist())
vals = re.findall(r"\d{4}", keys)
keys = keys.split(";")
budget = budget.rename(index=dict(zip(keys,vals)))

# reverse order
budget = budget.iloc[::-1] # reverse order
# prettify index
budget.index.name = "Year"
budget.index = budget.index.astype(int)

# subset from 2006 onwards
budget = budget.loc[2006:]
# remove commas in 2006-2018
budget.loc[2006:2018] = budget.loc[2006:2018].apply(
    lambda series: series.str.replace(",", "").astype(int)
)

"""Budget Feature Engineering

    New Columns
    ---------------
    Combined: State + Federal Budget
    Inflation: combined budget adjusted for inflation to 2020 using CPI
    Scaled 1e9: inflation scaled to the billions
"""
# combined budget
budget["Combined"] = budget["State"] + budget["Federal"]
budget = budget.drop(columns=["State", "Federal"])
# adjust for inflation
budget["Inflation"] =[
    budget["Combined"].loc[year]*cpi["Annual"].loc[2020]/cpi["Annual"].loc[year]
    for year in budget.index
]
# scale to the billions
budget["Scaled 1e9"] = np.round(budget["Inflation"] / 1e9, 2)


"""ELA Helper Functions
"""
def fill_school_name(df):
    """bfills the first valid school name in a df for every school.
    """
    res = []
    for dbn in df["DBN"].unique():
        sub = df[df["DBN"] == dbn]
        temp = sub[(sub["Demographic"] == "All Students") & (sub["Grade"] == "All Grades")]
        idx = temp["School Name"].first_valid_index()
        if idx is not None:
            name = temp["School Name"].loc[idx]
            sub["School Name"] = sub["School Name"].fillna(name)
        
        res.append(sub)
    return pd.concat(res)

def flatten_scores(df):
    """Flattens the score cols between 2006-2012 and 2013-2018
    """
    drops = []
    
    for i in range(1, 5):
        df[f"Level {i} #"] = df[f"Level {i} #"].fillna(df[f"Num Level {i}"])
        df = df[ ~df[f"Level {i} #"].str.contains("s")]
        drops.append(f"Num Level {i}")
        
        df[f"Level {i} %"] = df[f"Level {i} %"].fillna(df[f"Pct Level {i}"])
        df = df[ ~df[f"Level {i} %"].str.contains("s")]
        drops.append(f"Pct Level {i}")

        
    df["Level 3+4 #"] = df["Level 3+4 #"].fillna(df["Num Level 3 and 4"])
    df = df[ ~df["Level 3+4 #"].str.contains("s") ] 
    drops.append("Num Level 3 and 4")
    
    df["Level 3+4 %"] = df["Level 3+4 %"].fillna(df["Pct Level 3 and 4"])
    df = df[ ~df["Level 3+4 %"].str.contains("s") ] 
    drops.append("Pct Level 3 and 4")

    return df.drop(columns = drops)

def flatten_demographics(df, common):
    """Flattens the demographic vals between 2006-2012 and 2013-2018
    """
    df["Demographic"] = df["Demographic"].fillna(df["Category"])
    # keep only common demos
    df = df[df["Demographic"].isin(common)]
    df = df[df["Demographic"] != "Not SWD"]
    return df.drop(columns = ["Category"])

def set_typing(df):
    """sets the type of each series in the df
    """
    # float
    float_cols = ["Number Tested", "Level 3+4 #", "Level 3+4 %", "Year"]
    float_cols += [f"Level {i} #" for i in range(1, 5)]
    float_cols += [f"Level {i} %" for i in range(1, 5)]

    # str cols
    str_cols = ["Grade", "Demographic", "School Name", "Boro"]

    # type mapping
    type_mapping = {col: "float" for col in float_cols}
    type_mapping |= {col: "str" for col in str_cols}

    return df.astype(type_mapping)

def formatName(df):
    """Removes all punctuation from schools and makes uppercase
    """
    name_mapping = {n:"" for n in df["School Name"].unique()}
    
    for key in name_mapping.keys():
        formatted = re.sub(r"[^\w\d /]+", "", key)
        formatted = re.sub(r" 0", " ", formatted)
        name_mapping[key] = formatted.upper()
        
    return name_mapping

def createAllGrades(df):
    """Creates an All Grades value for grades in the df
    """
    res = []

    for school in df["School Name"].unique():
        school_df = df[df["School Name"] == school]

        idx = []
        for demo in school_df["Demographic"].unique():
            # get df
            data_df = school_df[school_df["Demographic"] == demo]

            # get any row
            idx = [False] * len(data_df)
            idx[0] = True
            new_row = data_df.loc[idx].copy()

            # fill in the new row
            new_row["Grade"] = "All Grades"
            new_row["Number Tested"] = data_df["Number Tested"].sum()

            for i in range(1, 5):
                new_row[f"Level {i} #"] = data_df[f"Level {i} #"].sum()
                new_row[f"Level {i} %"] = np.round(new_row[f"Level {i} #"] / new_row["Number Tested"] * 100, 2)

            new_row["Level 3+4 #"] = new_row["Level 3 #"] + new_row["Level 4 #"]
            new_row["Level 3+4 %"] = np.round(new_row["Level 3+4 #"] / new_row["Number Tested"] * 100, 2)

            # append to res
            res.append(new_row)
    
    return pd.concat(res)

"""2006-2018 ELA Cleaning

    Raw 2006-2012 Columns
    -----------------
    DBN: DBN of the school
    Grade: grade tested; also includes "All Grades"
    Year: year
    Demographic: demographic of students tested
    Number Tested: total # of students tested
    Mean Scale Score: mean scale score for demographic
    Num Level X: # of lvl X scores where X is 1-4
    Pct Level X: % of lvl X scores where X is 1-4
    Num Level 3 and 4: # of lvl 3-4 scores
    Pct Level 3 and 4: % of lvl 3-4 scores
    
    Raw 2013-2018 Columns
    -----------------
    DBN: DBN of the school
    School Name: name of the school
    Grade: grade tested; also includes "All Grades"
    Year: year
    Category: demographic of students tested
    Number Tested: total # of students tested
    Mean Scale Score: mean scale score for demographic
    Level X #: # of lvl X scores where X is 1-4
    Level X %: % of lvl X scores where X is 1-4
    Level 3+4 #: # of lvl 3-4 scores
    Level 3+4 %: % of lvl 3-4 scores
    
    Clean 2006-2018 Columns
    -----------------
    DBN: DBN of the school
    School Name: name of the school
    Grade: grade tested; also includes "All Grades"
    Year: year
    Demographic: demographic of students tested
    Number Tested: total # of students tested
    Level X #: # of lvl X scores where X is 1-4
    Level X %: % of lvl X scores where X is 1-4
    Level 3+4 #: # of lvl 3-4 scores
    Level 3+4 %: % of lvl 3-4 scores
"""
# read and merge
ela_2006_2012 = pd.concat([ pd.read_csv(prefix + csv) for csv in csvs[2:5] ])
ela_2013_2018 = pd.read_csv(prefix + csvs[5])
ela_2006_2018 = pd.concat([ela_2006_2012, ela_2013_2018])

# get common demographics
common = set(ela_2006_2012["Demographic"].unique()) & set(ela_2013_2018["Category"].unique())
# reshaping
ela_2006_2018 = flatten_scores(ela_2006_2018)
ela_2006_2018 = flatten_demographics(ela_2006_2018, common)
ela_2006_2018 = fill_school_name(ela_2006_2018)

# any leftover nan schools have been closed/merged
ela_2006_2018.loc[
    ela_2006_2018["School Name"].isnull(), "School Name"
] = "Closed or Merged"

# no need for scale scores
ela_2006_2018 = ela_2006_2018.drop(columns = ["Mean Scale Score"]).reset_index(drop = True)

"""2006-2018 ELA Feature Engineering

    New Columns
    ---------------
    Boro: boro of the school
"""
# add boros
mapping = {
    "K": "Brooklyn",
    "Q": "Queens",
    "M": "Manhattan",
    "X": "Bronx",
    "R": "Staten Island"
}
ela_2006_2018["Boro"] = ela_2006_2018["DBN"].str.get(2).map(mapping)

"""2019 ELA Cleaning

    Raw 2019 Columns
    -----------------
    SY_END_DATE: school year end date
    NRC_DESC: school demograhpic groupings;
        one of [
            nan, 'NYC', 'Buffalo, Rochester, Yonkers, Syracuse',
           'Urban-Suburban High Needs', 'Rural High Needs', 'Average Needs',
           'Low Needs', 'Charters'
       ]
    NRC_CODE: nrc code
    COUNTY_CODE: county code
    COUNTY_DESC: name of county
    BEDSCODE: beds code for the school
    NAME: school name
    ITEM_SUBJECT_AREA: subject for tests; one of ["ELA", "Math"]
    ITEM_DESC: Grades tested for ITEM_SUBJECT_AREA
    SUBGROUP_CODE: demographic code
    SUBGROUP_NAME: demographic
    TOTAL_TESTED: # of students tested
    LX_COUNT: # of lvl X scores where X is 1-4
    LX_PCT: % of lvl X scores where X is 1-4
    L2-L4 PCT: % of lvl 2-4 scores
    L3-L4 PCT: % of lvl 3-4 scores
    MEAN_SCALE_SCORE: mean scale score
    
    Clean 2019 Columns
    -----------------
    COUNTY_DESC: name of county
    NAME: school name
    ITEM_DESC: Grades tested for ITEM_SUBJECT_AREA
    SUBGROUP_NAME: demographic
    TOTAL_TESTED: # of students tested
    LX_COUNT: # of lvl X scores where X is 1-4
    LX_PCT: % of lvl X scores where X is 1-4
    L2-L4 PCT: % of lvl 2-4 scores
    L3-L4 PCT: % of lvl 3-4 scores
"""
ela_2019 = pd.read_csv(prefix + csvs[6])
# cols to drop
drops = [
    "NRC_CODE", "NRC_DESC", "BEDSCODE", "SUBGROUP_CODE",
    "MEAN_SCALE_SCORE", "L2-L4_PCT", "SY_END_DATE"
]

# ELA grades only
ela_2019 = ela_2019[ ela_2019["ITEM_SUBJECT_AREA"] == "ELA"]
drops.append("ITEM_SUBJECT_AREA")
# NYC counties only
nyc_counties = ["BRONX", "KINGS", "QUEENS", "NEW YORK", "RICHMOND"]
ela_2019 = ela_2019[ ela_2019["COUNTY_DESC"].isin(nyc_counties)]
drops.append("COUNTY_CODE")

# drop cols with county/district summary
ela_2019 = ela_2019[ ~ela_2019["NAME"].str.contains("COUNTY")]
ela_2019 = ela_2019[ ~ela_2019["NAME"].str.contains("GEOGRAPHIC")]

# drop cols with empty score
for i in range(1, 5):
    ela_2019 = ela_2019[ ~ela_2019[f"L{i}_COUNT"].str.contains("-")]
    ela_2019 = ela_2019[ ~ela_2019[f"L{i}_PCT"].str.contains("-")]
    
    # cast to num and convert to int while we're here
    ela_2019[f"L{i}_COUNT"] = ela_2019[f"L{i}_COUNT"].astype(int)
    ela_2019[f"L{i}_PCT"] = ela_2019[f"L{i}_PCT"].str[:-1].astype(int)

# convert these to nums while we're here
ela_2019["L3-L4_PCT"] = ela_2019["L3-L4_PCT"].str[:-1].astype(int)
ela_2019["TOTAL_TESTED"] = ela_2019["TOTAL_TESTED"].astype(int)
# drop cols
ela_2019 = ela_2019.drop(columns=drops)

"""2019 ELA Feature Engineering

['Boro', 'School Name', 'Grade', 'Demographic', 'Number Tested',
       'Level 1 #', 'Level 1 %', 'Level 2 #', 'Level 2 %', 'Level 3 #',
       'Level 3 %', 'Level 4 #', 'Level 4 %', 'Level 3+4 %', 'Year',
       'Level 3+4 #']
    
    2019 Columns
    -----------------
    Boro: boro of the school
    School Name: name of the school
    Grade: grade tested; also includes "All Grades"
    Year: year
    Demographic: demographic of students tested
    Number Tested: total # of students tested
    Level X #: # of lvl X scores where X is 1-4
    Level X %: % of lvl X scores where X is 1-4
    Level 3+4 #: # of lvl 3-4 scores
    Level 3+4 %: % of lvl 3-4 scores
"""
# add boros
mapping = {
    "KINGS": "Brooklyn",
    "QUEENS": "Queens",
    "NEW YORK": "Manhattan",
    "BRONX": "Bronx",
    "RICHMOND": "Staten Island"
}
ela_2019["COUNTY_DESC"] = ela_2019["COUNTY_DESC"].map(mapping)

# level rename mapping
level_mapping = [(f"L{i}_COUNT", f"Level {i} #") for i in range(1, 5)]
level_mapping += [(f"L{i}_PCT", f"Level {i} %") for i in range(1, 5)]
level_mapping.append(("L3-L4_PCT", "Level 3+4 %"))
# create col mapping
rename_mapping = dict(level_mapping)
rename_mapping |= {
    "TOTAL_TESTED": "Number Tested",
    "NAME": "School Name",
    "ITEM_DESC": "Grade",
    "COUNTY_DESC": "Boro",
    "SUBGROUP_NAME": "Demographic"
}
# rename
ela_2019 = ela_2019.rename(columns = rename_mapping)

# add year
ela_2019["Year"] = 2019

# flatten demographics
ela_2019.loc[ela_2019["Demographic"] == "Black or African American", "Demographic"] = "Black"
ela_2019.loc[ela_2019["Demographic"] == "Hispanic or Latino", "Demographic"] = "Hispanic"
ela_2019.loc[ela_2019["Demographic"] == "Asian or Pacific Islander", "Demographic"] = "Asian"
ela_2019.loc[ela_2019["Demographic"] == "Students with Disabilities", "Demographic"] = "SWD"

# flatten grade
ela_2019["Grade"] = ela_2019["Grade"].apply(lambda x: re.findall(r"\d", x)[0])
# all grades
ela_2019 = pd.concat([ela_2019, createAllGrades(ela_2019)])

# fill 3+4 count
ela_2019["Level 3+4 #"] = ela_2019["Level 3 #"] + ela_2019["Level 4 #"]

"""2020 ELA

    There is no data for the 2020 ELAs because it was cancelled due to COVID-19.
    All data for 2020 is interpolated when graphed.
"""