# Examining the effect of policy & federal spending on ELA scores

## todo

1. ~~find which CSVs to download~~  
2. ~~download data~~  
3. ~~clean data~~  
--> reformat into cleaner data for chart usage  
--> create single csv of clean data  
4. get frontend / chart working    
--> ~~create website~~ --> luiscollado.com/csci778-final-project  
--> ~~`chart.js` library~~  
--> ~~get js to read csv~~  

## Andy Mina - changelog

- W9: research data sets to be used, the columns that need to be cleaned or restructured, and add them to the repo. create the main `proj.py` file and add datasets to the code
  - Time spent: ~3 hours
- W10: reformat and restructure the budget data to create nwe columns. left documentation regarding what the feature engineering was. inflation is scaled down to the billions. lots of pandas review
  - Time spent: ~3 hours
- W11: define helper functions to flatten demographics across the varied data. 2006-2012 has a different column/entry format than 2013-2018.
  - Time spent: ~3.5 hours
- W12: clean the merged 2006-2018 dataset to be consistent. left some documentation on the shape of the old datasets and the final shape of the new data sets
  - Time spent: ~4 hours
- W13: clean data for 2019 ELA to match the same format as the existing 2006-2018 data. also add in school boro. also did some work to retrieve school names from google maps API by searchign the school code and getting the boros for missing schools.
  - Time spent: ~6 hours
- W14: using the merged data, generate the corresponding graphs and animations. this required some good research into how matplotlib handles internal graphing.
  - Time spent: ~2.5 hours

## Luis Collado - changelog
