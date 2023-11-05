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