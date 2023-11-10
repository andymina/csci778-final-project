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



"""2021 ELA Cleaning

    Raw 2021 ELA Columns
    -----------------
    SY_END_DATE: school year end date
    BEDSCODE: beds code for the school
    NAME: school name
    ITEM_SUBJECT_AREA: subject for tests; one of ["ELA", "Math"]
    ITEM_DESC: Grades tested for ITEM_SUBJECT_AREA
    SUBGROUP_CODE: demographic code
    SUBGROUP_NAME: demographic
    TOTAL_ENROLLED: # of students enrolled
    TOTAL_TESTED: # of students tested
    TOTAL_NOT_TESTED: # of students not tested
    LX_COUNT: # of lvl X scores where X is 1-4
    LX_PCT: % of lvl X scores where X is 1-4
    L3-L4 PCT: % of lvl 3-4 scores
    MEAN_SCALE_SCORE: mean scale score
    
    Raw NYC 2021 ELA Columns
    -----------------
    SCHOOL YEAR END DATE: school year end date
    BEDS CODE: beds code for the school
    NAME: school name
    SUBJECT: Grade and subject tested on
    TOTAL ENROLLED: # of students enrolled
    TOTAL NOT TESTED: # of students not tested
    TOTAL TESTED: # of students tested
    LEVEL X COUNT: # of lvl X scores where X is 1-4
    LEVEL X PCT: % of lvl X scores where X is 1-4
    LEVEL 3-4 PCT: % of lvl 3-4 scores
    MEAN SCALE SCORE: mean scale score
    
    Clean 2021 ELA Columns
    -----------------
    NAME: school name
    ITEM_DESC: Grades tested for ITEM_SUBJECT_AREA
    SUBGROUP_NAME: demographic
    TOTAL_TESTED: # of students tested
    LX_COUNT: # of lvl X scores where X is 1-4
    LX_PCT: % of lvl X scores where X is 1-4
    L3-L4 PCT: % of lvl 3-4 scores
"""
ela_2021 = pd.read_csv(prefix + csvs[7])
nyc_ela_2021 = pd.read_csv(prefix + csvs[8])

# store columns to drop
drops = [
    "BEDSCODE", "TOTAL_ENROLLED", "TOTAL_NOT_TESTED",
    "MEAN_SCALE_SCORE", "SY_END_DATE", "SUBGROUP_CODE"
]

# only nyc schools
nyc_schools = set(nyc_ela_2021["NAME"].unique())
ela_2021 = ela_2021[ela_2021["NAME"].isin(nyc_schools)]
# ELA grades only
ela_2021 = ela_2021[ela_2021["ITEM_SUBJECT_AREA"] == "ELA"]
drops.append("ITEM_SUBJECT_AREA")
# drop cols with county/district summary
ela_2021 = ela_2021[ ~ela_2021["NAME"].str.contains("COUNTY")]
ela_2021 = ela_2021[ ~ela_2021["NAME"].str.contains("GEOGRAPHIC")]

# drop rows with empty count/pct
for i in range(1, 5):
    ela_2021 = ela_2021[ ~ela_2021[f"L{i}_COUNT"].str.contains("-") ]
    ela_2021 = ela_2021[ ~ela_2021[f"L{i}_PCT"].isnull() ]
    
    # cast to num
    ela_2021[f"L{i}_COUNT"] = ela_2021[f"L{i}_COUNT"].astype(int)
    ela_2021[f"L{i}_PCT"] = ela_2021[f"L{i}_PCT"].str[:-1].astype(int)
    
# flatten demographic
groups = [
    "All Students", "Black or African American", "Asian or Pacific Islander",
    "White", "Hispanic or Latino", "Students with Disabilities"
]
ela_2021 = ela_2021[ela_2021["SUBGROUP_NAME"].isin(groups)]

# drop the cols
ela_2021 = ela_2021.drop(columns = drops)


"""2021 ELA Feature Engineering
       
    2021 Columns
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
# add year
ela_2021["Year"] = 2021
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
ela_2021 = ela_2021.rename(columns = rename_mapping)

# flatten grade
ela_2021["Grade"] = ela_2021["Grade"].apply(lambda x: re.findall(r"\d", x)[0])
# flatten demographics
ela_2021.loc[ela_2021["Demographic"] == "Black or African American", "Demographic"] = "Black"
ela_2021.loc[ela_2021["Demographic"] == "Hispanic or Latino", "Demographic"] = "Hispanic"
ela_2021.loc[ela_2021["Demographic"] == "Asian or Pacific Islander", "Demographic"] = "Asian"
ela_2021.loc[ela_2021["Demographic"] == "Students with Disabilities", "Demographic"] = "SWD"

# find out which schools 2019 and 2021 have in common/different
schools_2021 = set(ela_2021["School Name"].unique())
schools_2019 = set(ela_2019["School Name"].unique())
common = list(schools_2021.intersection(schools_2019))
diff = list(schools_2021.difference(schools_2019))

# retrieve boros from Google Maps
url = "https://maps.googleapis.com/maps/api/place/textsearch/json?"
payload = {}
headers = {}
boros = []

for name in diff:
    query = urlencode({"query": name, "key": "AIzaSyCwF1cFyjLd6Txrczy_H2jZQj6NUWd4OZ0"})
    response = requests.request(
        "GET", url+query, headers = headers, data = payload
    )
    res = json.loads(response.text)["results"]
    
    if len(res) > 0:
        res = res[0]
        county = re.findall(r", [\w ]+,", res["formatted_address"])[0][2:-1].upper()
        boros.append(county)
    else:
        boros.append(None)

        # set JHS 291 ROLAND HAYES manually
boros[boros.index(None)] = "BROOKLYN"
# add boros of missing schools to dataframe
for i in range(len(diff)):
    ela_2021.loc[ela_2021["School Name"] == diff[i], "Boro"] = boros[i]
    
# add boros of common schools
for school in common:
    boro = ela_2019[ela_2019["School Name"] == school]["Boro"].unique()[0]
    ela_2021.loc[ela_2021["School Name"] == school, "Boro"] = boro
    
# create boro rename mapping
rename_mapping = {
    "QUEENS": "Queens",
    "NEW YORK": "Manhattan",
    "BRONX": "Bronx",
    "BROOKLYN": "Brooklyn",
    "STATEN ISLAND": "Staten Island"
}

# grab boros in df
boros_in_df = set(ela_2021["Boro"].unique())
boros_in_df.difference_update(rename_mapping.keys())
boros_in_df.difference_update([s.title() for s in rename_mapping.keys()])
boros_in_df.remove("Manhattan")
# set the renaming boros to queens by hand
rename_mapping |= {b:"Queens" for b in boros_in_df}
# rename
ela_2021["Boro"] = ela_2021["Boro"].replace(to_replace = rename_mapping)

# all grades
ela_2021 = pd.concat([ela_2021, createAllGrades(ela_2021)])

# add prof # and %
ela_2021["Level 3+4 #"] = ela_2021["Level 3 #"] + ela_2021["Level 4 #"]
ela_2021["Level 3+4 %"] = ela_2021["Level 3+4 %"].str[:-1]



"""Merge 2006-2021
"""
ela = pd.concat([ela_2006_2018, ela_2019, ela_2021]).reset_index(drop = True)
# set typing
ela = set_typing(ela)
# format everyone's name
ela["School Name"] = ela["School Name"].replace(to_replace=formatName(ela))
# fix typo by hand
typo = "JOHN W LAVELLE PREPARATORY CHARTER SCHOO"
ela.loc[ela["School Name"] == typo, "School Name"] = typo + "L"

# redo %
for i in range(1, 5):
    ela[f"Level {i} %"] = np.round(ela[f"Level {i} #"] / ela["Number Tested"] * 100, 2)
ela["Level 3+4 %"] = np.round(ela["Level 3+4 #"] / ela["Number Tested"] * 100, 2)

# fill DBN with empty
ela["DBN"] = ela["DBN"].fillna("00N000")



"""Analysis Helper Code
"""
def createProfPcts(df, cols):
    """Returns a df containing the proficiency %s by year grouped by `cols`.
    """
    g = df.groupby(cols, as_index = True)
    pct = g["Level 3+4 #"].sum() / g["Number Tested"].sum() * 100
    if len(cols) > 1:
        pct = pct.unstack().transpose()
        pct.columns.name = None
        
    pct = pct.fillna(0)
    
    return pct

def createPctAni(
    data, lines,
    title="", xlabel="", ylabel="",
    start_year=2006, end_year=2021
):
    """Returns a matplotlib.figure and matplotlib.axis containing a line graph of the given data.
    """
    # update animation frame
    def update(frame, ax, lines, data):
        for line in lines:
            year = frame + start_year
            if year != 2020:
                xdata, ydata = line.get_data() # get x, y
                if year == 2006:
                    xdata = []
                    ydata = []
                xdata.append(year) # update x 
                ydata.append(data[line.get_label()].loc[year]) # update y
                line.set_data(xdata, ydata) # set data

    # create the plot
    fig, ax = plt.subplots()
    ax.grid()
    # set title
    ax.set_title(title)
    # set x axis
    ax.set_xlim(start_year, end_year)
    ax.set_xlabel(xlabel)
    # set y axis
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)

    # render the lines
    for line in lines:
        ax.add_line(line)
    
    # set up legend
    ax.legend(handles = lines, fontsize = "x-small")
    total_frames = end_year - start_year + 1

    # create animation
    ani = FuncAnimation(
        fig, update, frames = total_frames, interval = 1000,
        fargs = (ax, lines, data),
        repeat = True
    )
    
    return fig, ani

def compute_r_line(xes, yes):
    sd_x = np.std(xes)
    sd_y = np.std(yes)
    r = np.corrcoef(xes, yes)[0][1]
    m = r * sd_y / sd_x
    b = yes[0] - m * xes[0]
    return r, m, b

def createBudgetPlot(xdata, ydata, label, color):
    """Creates a lmplot of the given data
    """
    fig, ax = plt.subplots()
    
    # get data
    corr, m, b = compute_r_line(xdata, ydata)
    corr = np.round(corr, 2)
    m = np.round(m, 2)
    b = np.round(b, 2)

    # plot data
    ax.scatter(xdata, ydata, color = color)
    
    xes = np.arange(10, 14.5, .5)
    yes = [m*x + b for x in xes]
    r_line = Line2D(xes, yes, label = f"y = {m}x + {b}\n corr = {corr}")
    # plot r line
    ax.add_line(r_line)#(xes, yes, label = f"y = {m}x + {b}\n corr = {corr}")

    ax.set(
        xticks = xes,
        xlabel = "Budget ($ in billions)",
        ylim = [0, 100],
        ylabel = "Percent of Proficient Students (%)",
        title = f"NY Education Budget vs. ELA Proficiency under {label}"
    )
    ax.xaxis.set_major_formatter("{x:1.2f}")
    ax.grid()
    ax.legend()
    
    return fig, ax

def createNYCProf(start_year=2006, end_year=2021):
    """Creates a df of NYC's overall proficiency
    """
    years = []
    pcts = []
    for year in range(start_year, end_year + 1):
        if year != 2020:            
            df = gen_ela[
                (gen_ela["Level 3+4 %"] >= 65) &
                (gen_ela["Year"] == year)
            ]

            res = len(df) / len(gen_ela[gen_ela["Year"] == year]) * 100
            pcts.append(np.round(res, 2))
            years.append(year)
    
    return pd.DataFrame(pcts, index=years, columns=["Proficient School %"])

# lines to represent each boro for graphing
boro_lines = [
    Line2D([], [], color = "darkorange", label = "Brooklyn"),
    Line2D([], [], color = "purple", label = "Queens"),
    Line2D([], [], color = "green", label = "Bronx"),
    Line2D([], [], color = "red", label = "Staten Island"),
    Line2D([], [], color = "blue", label = "Manhattan")
]

"""Graphs
"""
# General ELA Proficiency
gen_ela = ela[
    (ela["Grade"] == "All Grades") &
    (ela["Demographic"] == "All Students")
]
gen_ela_pcts = createProfPcts(gen_ela, ["Boro", "Year"])
gen_fig, gen_ani = createPctAni(
    gen_ela_pcts, [dc(x) for x in boro_lines], title="Overall ELA Proficiency",
    xlabel="Year", ylabel="Percent of Proficient Students (%)"
)
gen_ani.save("assets/Overall-Proficiency.gif")

# Children of Color ELA proficiency
coc_ela = ela[
    (ela["Grade"] == "All Grades") &
    (ela["Demographic"] != "White")
]
coc_ela_pcts = createProfPcts(coc_ela, ["Boro", "Year"])
coc_fig, coc_ani = createPctAni(
    coc_ela_pcts, [dc(x) for x in boro_lines], title="Children of Color ELA Proficiency",
    xlabel="Year", ylabel="Percent of Proficient Students (%)"
)
coc_ani.save("assets/Children-Proficiency.gif")

# SWD proficiency
swd_ela = ela[
    (ela["Grade"] == "All Grades") &
    (ela["Demographic"] == "SWD")
]
swd_ela_pcts = createProfPcts(swd_ela, ["Boro", "Year"])
swd_fig, swd_ani = createPctAni(
    swd_ela_pcts, [dc(x) for x in boro_lines], title="SWD ELA Proficiency",
    xlabel="Year", ylabel="Percent of Proficient Students (%)"
)
swd_ani.save("assets/SWD-Proficiency.gif")

# create y data
year_pcts = createProfPcts(gen_ela, ["Year"])

# nclb budget
nclb_x = budget["Scaled 1e9"].loc[2006: 2014].tolist()
nclb_y = year_pcts.loc[2006:2014].tolist()
nclb_fig, nclb_ax = createBudgetPlot(nclb_x, nclb_y, "NCLB", "red")
nclb_fig.savefig("assets/NCLB-Budget.png")

# essa vs budget
essa_x = budget["Scaled 1e9"].loc[2015:2019].tolist()
essa_y = year_pcts.loc[2015:2019].tolist()
essa_fig, essa_ax = createBudgetPlot(essa_x, essa_y, "ESSA", "green")
essa_fig.savefig("assets/ESSA-Budget.png")


# NYC Overall Proficiency
nyc_prof = createNYCProf(2006, 2021)
nyc_fig, nyc_ani = createPctAni(
    nyc_prof, [Line2D([], [], color="red", label="Proficient School %")],
    title="NYC Schools with a Proficiency Rate over 65%",
    xlabel="Year", ylabel="Percent of Proficient Schools (%)"
)
nyc_ani.save("assets/NYC-Proficiency.gif")

# Common Core
years = [y for y in range(2013, 2019)]
grade_3 = ela[
    (ela["Demographic"] == "All Students") &
    (ela["Grade"] == "3")
]
grade_4 = ela[
    (ela["Demographic"] == "All Students") &
    (ela["Grade"] == "4")
]
grade_5 = ela[
    (ela["Demographic"] == "All Students") &
    (ela["Grade"] == "5")
]

# # of schools with no proficients 3rd grades from 2013-2017
grade3_bars = [len(grade_3[
    (grade_3["Year"] == year) &
    (grade_3["Level 3+4 #"] == 0)
]) for year in years]
# # of schools with no proficients 3rd grades from 2013-2017
grade4_bars = [len(grade_4[
    (grade_4["Year"] == year) &
    (grade_4["Level 3+4 #"] == 0)
]) for year in years]
# # of schools with no proficients 3rd grades from 2013-2017
grade5_bars = [len(grade_5[
    (grade_5["Year"] == year) &
    (grade_5["Level 3+4 #"] == 0)
]) for year in years]

# graph
width = 0.25
cc_bar_fig, cc_bar_ax = plt.subplots()
cc_bar_ax.bar([y - width for y in years], grade3_bars, width, label="Grade 3")
cc_bar_ax.bar([y for y in years], grade4_bars, width, label="Grade 4")
cc_bar_ax.bar([y + width for y in years], grade5_bars, width, label="Grade 5")

cc_bar_ax.legend()
cc_bar_ax.set(
    title="# of Schools with No Proficient Students in Grade",
    xlabel="Year",
    ylabel="# of Schools",
    ylim=[0, 10]
)
cc_bar_fig.savefig("assets/Common-Core-Bar.png")

# NYC schools with < 20% prof
years = gen_ela["Year"].unique()
nyc_bars = [
    len(gen_ela[(gen_ela["Level 3+4 %"] < 20) & (gen_ela["Year"] == year)])
    for year in years
]

# create graph
nyc_prof_fig, nyc_prof_ax = plt.subplots()
nyc_prof_ax.bar(years, nyc_bars, width=0.5)

# label graph
nyc_prof_ax.set(
    title="NYC Schools with < 20% Proficiency",
    xlabel="Year",
    ylabel="# of Schools"
)
nyc_prof_fig.savefig("assets/NYC-Prof-Bar.png")

years = [y for y in range(2006, 2013)]
schools_closed = [
    len(ela[(ela["School Name"] == "CLOSED OR MERGED") & (ela["Year"] == year)])
    for year in years
]

# create graph
school_fig, school_ax = plt.subplots()
school_ax.bar(years, schools_closed, width=0.5)

# label graph
school_ax.set(
    title="# of NYC Schools that Have Since Been Closed/Merged from 2006-2012",
    xlabel="Year",
    ylabel="# of Schools"
)
school_fig.savefig("assets/NYC-Close-Bar.png")