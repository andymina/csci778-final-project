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