import pandas as pds # pylint: disable=import-error
import numpy as np # pylint: disable=import-error

PATH_TO_GOOGLE_CSV = '/home/matt/data/covid_community_mobility/Global_Mobility_Report_11_27_20.csv'
PATH_TO_NYTIME_CSV = '/home/matt/data/covid_community_mobility/us-states.csv'

def rolling_window(a, window_size):
    shape = (a.shape[0] - window_size + 1, window_size) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def get_df(state):
    df_google = pds.read_csv(PATH_TO_GOOGLE_CSV, low_memory=False)
    df_nytime = pds.read_csv(PATH_TO_NYTIME_CSV)
    
    specific_google = (df_google.loc[df_google['sub_region_1'] == state].sort_values('date'))[['sub_region_1', 'sub_region_2', 'metro_area', 'date', 
        'retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline',
        'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline']]
    
    state_google = ((specific_google.loc[pds.isna(specific_google['sub_region_2'])].sort_values('date'))[['date', 
        'retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline',
        'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline']]
                    .set_index('date')).loc['2020-03-02':]

    state_nytime = ((((df_nytime.loc[df_nytime['state'] == state].sort_values('date'))[['date', 'new_cases']])
                .set_index('date')).loc[:'2020-11-24'])

    return state_google, state_nytime