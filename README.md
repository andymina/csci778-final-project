# Examining the effect of policy & federal spending on ELA scores

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

- W9  ~2.5 hrs
  - learn how to deal with more complicated CSVs in pandas
  - refresh regex knowledge for renaming index
  - handle the budget and consumer price index file
  - get simple, clean columns of state + federal budget
- W10  ~2 hrs
  - learned how to work with dataframes - Andy had a bit more experience than me. 
  - wrote 2 functions that matched andy's function-based focus, fill_school_name and flatten_scores. 
  - spent a lot of time troubleshooting .contains() on a string versus on a dataframe's .str value
- W11  ~2 hrs
  - regex reviewing for name formatting
  - a lot of time spent on regexr
  - i was better with dataframes by now
  - wrote formatName() to take a dataframe and have all schools be uppercase, no punctuation
  - createAllGrades() which adds a new column to the dataframe that combines performance in all grades
- W12  ~2 hrs
  - read through 2019 ELA csv and document it (!!!!!)
  - determine which columns need to be included in the code/functions
  - begin reading ELA csv, review pandas methods
  - force code to adhere to previous style choices
- W13  ~2 hrs
  - same thing as W11 except with 2021 ELAs becase no 2020 ELAs, feature engineer 2021 ELAs (was able to reuse code/ideas)
  - review set/unique etc. to handle schools closing and/or merging
  - buy luiscollado.com and look into how to upload code through sftp from codespaces
  - determine how to best represent charts in javascript
  - get basic website up and running
  - html, css, js review!
- W14  ~4.5 hrs
  - get chartjs proof of concept running with budget code
  - finally merge 2006-2021 cleaned data columns 
  - learn how to make animations in matplotlib (!!!!)
  - write percentage of proficient schools animation
  - make + save + upload figures
  - move on to final website with more gifs and figures