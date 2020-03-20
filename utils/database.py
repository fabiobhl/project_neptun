import pandas as pd 
import numpy as np 
import ta
from binance.client import Client
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, CheckButtons, Button
import random
import warnings
import time
import pickle

class DataBase():

    def __init__(self, size, symbol):
        self.size = size
        self.symbol = symbol

        #these are datarows that will be normalized by the .pct_change method
        self.pctchange = ['open', 'high', 'low', 'close', 'volume', 'volume_nvi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl', 'volatility_bbw', 'volatility_kcc',
                          'volatility_kch', 'volatility_kcl', 'volatility_dcl', 'volatility_dch', 'trend_ema_fast', 'trend_ema_slow', 'trend_adx_pos', 'trend_adx_neg',
                          'trend_mass_index', 'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up', 'trend_aroon_down',
                          'trend_psar', 'trend_psar_up', 'trend_psar_down', 'momentum_rsi', 'momentum_uo', 'momentum_kama']
        #these are datarows that will be normalized by the .diff method
        self.diff = ['volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em', 'volume_sma_em', 'volume_vpt', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff',
                     'trend_trix', 'trend_cci', 'trend_dpo', 'trend_kst', 'trend_kst_sig', 'trend_kst_diff', 'momentum_mfi', 'momentum_tsi', 'momentum_stoch','momentum_stoch_signal',
                     'momentum_wr', 'momentum_ao', 'momentum_roc']
        
        #download the data
        self.raw_data = self.download()
        #add the tas
        self.ta_data = self.addtas()
        #derive the data
        self.derive_data = self.derive()
        #scale the data
        self.scaled_data = self.scale()

    #method to download the data
    def download(self):
        #initialize the client
        client = Client(api_key=keys.key, api_secret=keys.secret)
        
        #download the data
        print('downloading data...')
        raw_data = client.get_historical_klines(symbol=self.symbol, interval='1m', start_str=f'{self.size + 2000} minutes ago UTC')
        data = pd.DataFrame(raw_data)

        #clean the dataframe
        data = data.astype(float)
        data.drop(data.columns[[7,8,9,10,11]], axis=1, inplace=True)
        data.rename(columns = {0:'open_time', 1:'open', 2:'high', 3:'low', 4:'close', 5:'volume', 6:'close_time'}, inplace = True)

        #set time as index & drop open_time
        data['close_time'] += 1
        data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')
        data.set_index('close_time', inplace=True)
        data.drop('open_time', axis=1, inplace=True)

        #set all values as floats
        data = data.astype(float)

        #check for nans
        if data.isna().values.any():
            print('Nan values in data, please discard this object and try again')

        #shorten dataframe to length of size+61 because we'll need extra lines for ta and normalizing 
        data = data.iloc[:self.size+61]
        print('downloaded the data')

        return data

        #function to add tas
    
    #method for adding the tas
    def addtas(self):
        #add the tas
        with warnings.catch_warnings(record=True):
            data = ta.add_all_ta_features(self.raw_data.copy(), open='open', high="high", low="low", close="close", volume="volume", fillna=True)
        #drop broken tas
        data.drop(['volatility_atr', 'trend_adx', 'others_dr', 'others_dlr', 'others_cr'], inplace=True, axis=1)
        #drop first 60 rows
        self.raw_data = self.raw_data[60:]
        data = data.iloc[60:]

        return data

    #calculate derivative like values
    def derive(self):
        data2 = pd.DataFrame()

        #pct_change
        for column in self.ta_data.columns:
            if column in self.pctchange:
                data2[column] = self.ta_data[column].copy().pct_change()
        #diff
        for column in self.ta_data.columns:
            if column in self.diff:
                data2[column] = self.ta_data[column].copy().diff()
        #no change
        for column in self.ta_data.columns:
            if (column not in self.diff) and (column not in self.pctchange):
                data2[column] = self.ta_data[column].copy()

        #delete first row
        self.raw_data = self.raw_data.iloc[1:]
        self.ta_data = self.ta_data.iloc[1:]
        data2 = data2.iloc[1:]

        return data2

        #scale data
    
    #scale data
    def scale(self):
        data = self.derive_data.copy()

        #reset the indices
        data.reset_index(inplace=True, drop=True)

        #initialize the scaler and fit it
        self.scaler = preprocessing.MinMaxScaler((-1,1))
        self.scaler.fit(data)

        #scale
        scaled_data = pd.DataFrame(self.scaler.transform(data))
        scaled_data.set_index(self.derive_data.index, inplace=True)
        scaled_data.columns = self.derive_data.columns

        return scaled_data

        #alternativa constructor
    
    #alternative constructor
    @staticmethod
    def from_database(databasepath):
        obj = pickle.load(open(databasepath, "rb"))
        if isinstance(obj, DataBase):
            return obj
        else:
            raise EnvironmentError('You tried to load a file which is not an instance of DataBase')

    #save the object
    def save(self, databasepath):
        pickle.dump(self, open(databasepath, "wb"))

