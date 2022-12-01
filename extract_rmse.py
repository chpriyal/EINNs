import pandas as pd
import numpy as np
from epiweeks import Week, Year
import math
# dtype = torch.float
import os
DAY_WEEK_MULTIPLIER = 7
SMOOTH_WINDOW = 21
WEEKS_AHEAD = 8
PAD_VALUE = -9
DAYS_IN_WEEK = 7
# daily
datapath = './data/covid-hospitalization-daily-all-state-merged_vEW202133.csv'
datapath_weekly = './data/covid-hospitalization-all-state-merged_vEW202133.csv'
county_datapath = f'./Data/Processed/county_data.csv'
EW_START_DATA = '202020'  # defaul if not provided
SMOOTH_MOVING_WINDOW = True
population_path = './data/table_population.csv'
path = '/storage/home/hcocice1/pchhatrapati3/EINNs/results/COVID/'

macro_features=[
    'retail_and_recreation_percent_change_from_baseline',
    'grocery_and_pharmacy_percent_change_from_baseline',
    'parks_percent_change_from_baseline',
    'transit_stations_percent_change_from_baseline',
    'workplaces_percent_change_from_baseline',
    'residential_percent_change_from_baseline',
    'apple_mobility',
    'cdc_hospitalized',
    'covidnet',
    'fb_survey_cli',
    'death_jhu_incidence',
    'positiveIncr',
    'negativeIncr',
    ]

regions = ['GA','CA','NY']

def convert_to_epiweek(x):
    return Week.fromstring(str(x))

def moving_average(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean().values

def load_df(region,ew_start_data,ew_end_data,temporal='daily'):
    """ load and clean data"""
    if temporal=='daily':
        df = pd.read_csv(datapath,low_memory=False)
    elif temporal=='weekly':
        df = pd.read_csv(datapath_weekly,low_memory=False)
    df = df[(df["region"] == region)]
    df['epiweek'] = df.loc[:, 'epiweek'].apply(convert_to_epiweek)
    # subset data using init parameters
    df = df[(df["epiweek"] <= ew_end_data) & (df["epiweek"] >= ew_start_data)]
    df = df.fillna(method="ffill")
    df = df.fillna(method="backfill")
    df = df.fillna(0)
    return df

def get_state_train_data(region,pred_week,ew_start_data=202036,temporal='daily'):
    """ get processed dataframe of data + target as array """
    # import data
    region = str.upper(region)
    pred_week=convert_to_epiweek(pred_week) 
    ew_start_data=convert_to_epiweek(ew_start_data)
    df = load_df(region,ew_start_data,pred_week,temporal)
    # smooth
    df.loc[:,'positiveIncr'] = moving_average(df.loc[:,'positiveIncr'].values,SMOOTH_WINDOW)
    df.loc[:,'death_jhu_incidence'] = moving_average(df.loc[:,'death_jhu_incidence'].values,SMOOTH_WINDOW)
    # select targets
    targets = df.loc[:,['death_jhu_incidence']].values
    # now subset based on input ew_start_data
    df = df[macro_features]
    return df, targets
nrmse1 = []
nrmse2 = []
nd = []
epiweeks = [202036,202038,202040,202042,202044,202046,202048,202050,202052,202101,202103,202105,202107,202109]
for feature in macro_features:
    
    max_pred = -1
    min_pred = 50000
    sumofpreds = 0
    sumofsqerror = 0
    sumoferr = 0
    n = 0
    for region in regions:
        d,t = get_state_train_data(region,str(202109+(8)))
        # print(t.shape)
        # exit()
        index = 0
        
        for epiweek in epiweeks:
            region_path = path+region
            files = os.listdir(region_path)
            files = [x for x in files if (str(epiweek) in x and feature in x)]
            # print(files,epiweek,feature)
            files = files[0]
            f = pd.read_csv(region_path+'/'+files)
            f = np.asarray(f['deaths'])
            pred_death = [np.mean(f[7*i:(i+1)*7]) for i in range(8)]
            actual_death = t[index*7:(index+8)*7]
            # print(index*7,(index+8)*7)
            actual_death = [np.mean(actual_death[7*i:(i+1)*7]) for i in range(8)]
            n+=8
            max_pred = max(max_pred, np.amax(pred_death))
            min_pred = min(min_pred, np.amin(pred_death))
            # print(pred_death)
            # print(actual_death)
            for ind,val in enumerate(pred_death):
                sumofpreds+=abs(val)
                sumofsqerror+=(val-actual_death[ind])**2
                # print((val-actual_death[ind])**2)
                sumoferr += abs(val-actual_death[ind])
            index+=2
    # print(n)
    # print(sumofsqerror)
    # print(sumofpreds)
    # print(max_pred)
    # print(min_pred)
    nrmse1.append(math.sqrt(sumofsqerror/n)/(sumofpreds/n)) #https://arxiv.org/pdf/2111.05199.pdf
    nrmse2.append(math.sqrt(sumofsqerror/n)/(max_pred-min_pred))
    nd.append(sumoferr/sumofpreds)
    
print(nrmse1)
print(nrmse2)
print(nd)
