import numpy as np # pylint: disable=import-error
import pandas as pds # pylint: disable=import-error
from hmmlearn import hmm # pylint: disable=import-error

PATH_TO_GOOGLE_CSV = '/home/matt/data/covid_community_mobility/Global_Mobility_Report_11_27_20.csv'
PATH_TO_NYTIME_CSV = '/home/matt/data/covid_community_mobility/covid-19-data/us-states.csv'

df_google = pds.read_csv(PATH_TO_GOOGLE_CSV, low_memory=False)
df_nytime = pds.read_csv(PATH_TO_NYTIME_CSV)
# print(df.sort_values('date'))
print(df_google.loc[df_google['country_region'] == 'United States'].sort_values('date'))