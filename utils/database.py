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

class SupervisedDataBase(DataBase):

    def __init__(self, size, symbol, labelwindowsize=60):
        super().__init__(size, symbol)
        self.labelwindowsize = labelwindowsize
        self.features = list(self.scaled_data.columns)
        self.label_counter = 0
        self.labelwindows = int(self.size/self.labelwindowsize)
        self.islabeled = False
        self.labeled_data = pd.DataFrame()

    #add labels
    def add_labels(self):
        """
        Add labels to the data by hand with the help of interactive matplotlib graph
        """
        #check if data is already labeled
        if self.islabeled:
            print('you already labeled your data')
            return

        #setup the dataframes
        data = self.scaled_data.copy()
        data2 = self.ta_data.copy()

        #add columns for labeling
        data['bsh'] = np.nan
        data['buy'] = np.nan
        data['sell'] = np.nan
        data['hold'] = np.nan

        #reset the indices
        data.reset_index(drop=True, inplace=True)
        data2.reset_index(drop=True, inplace=True)

        #setup the plotting
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        #add spaces for buttons
        axsave = plt.axes([0.4, 0.05, 0.2, 0.1])

        #callbackfunctions
        def updateplot():
            
            ax.cla()
            ax.grid()
            ax.plot(self.window2['close'])
            ax.plot(self.window['sell'], color='red', marker='o')
            ax.plot(self.window['buy'], color='green', marker='o')
            ax.set_title(f'{self.label_counter+1}/{self.labelwindows}')
            fig.canvas.draw()

        def updatebuy(eclick, erelease):
            start = int(round(eclick.xdata))
            end = int(round(erelease.xdata))

            for i in range(start, end+1, 1):
                self.window.loc[i, 'buy'] = None
                self.window.loc[i, 'sell'] = None
                self.window.loc[i, 'bsh'] = None
                self.window.loc[i, 'buy'] = self.window2.loc[i, 'close']
                self.window.loc[i, 'bsh'] = 1

            updateplot()

        def updatesell(eclick, erelease):
            start = int(round(eclick.xdata))
            end = int(round(erelease.xdata))

            for i in range(start, end+1, 1):
                self.window.loc[i, 'buy'] = None
                self.window.loc[i, 'sell'] = None
                self.window.loc[i, 'bsh'] = None
                self.window.loc[i, 'sell'] = self.window2.loc[i, 'close']
                self.window.loc[i, 'bsh'] = 2

            updateplot()

        def updatedelete(eclick, erelease):
            start = int(round(eclick.xdata))
            end = int(round(erelease.xdata))

            for i in range(start, end+1, 1):
                self.window.loc[i, 'sell'] = None
                self.window.loc[i, 'buy'] = None
                self.window.loc[i, 'bsh'] = None

            updateplot()
        
        def savefunction(event):
            
            #add holds to window
            for a in list(self.window.index):
                if pd.isna(self.window.loc[a, 'bsh']):
                    self.window.loc[a, 'bsh'] = 0
                    self.window.loc[a, 'hold'] = self.window2.loc[a, 'close']
            
            #append window to data3
            self.labeled_data = self.labeled_data.append(self.window, ignore_index=True)
            self.label_counter += 1

            #autosave the labeled data
            self.save('autosave.p')

            #check if were done
            if self.label_counter == self.labelwindows:
                plt.close()
                return

            #get the window
            self.window = data[self.label_counter*self.labelwindowsize:self.label_counter*self.labelwindowsize+self.labelwindowsize].copy()
            self.window.reset_index(drop=True, inplace=True)
            self.window2 = data2[self.label_counter*self.labelwindowsize:self.label_counter*self.labelwindowsize+self.labelwindowsize].copy()
            self.window2.reset_index(drop=True, inplace=True)

            updateplot()
        
        buySelector = RectangleSelector(ax, onselect=updatebuy, button=1)
        sellSelector = RectangleSelector(ax, onselect=updatesell, button=3)
        deleteSelector = RectangleSelector(ax, onselect=updatedelete, button=2)
        saver = Button(axsave, ['save'])
        saver.on_clicked(savefunction)

        self.window = data[self.label_counter*self.labelwindowsize:self.label_counter*self.labelwindowsize+self.labelwindowsize].copy()
        self.window.reset_index(drop=True, inplace=True)
        self.window2 = data2[self.label_counter*self.labelwindowsize:self.label_counter*self.labelwindowsize+self.labelwindowsize].copy()
        self.window2.reset_index(drop=True, inplace=True)

        updateplot()

        fig.show()
        plt.show()

        if self.label_counter == self.labelwindows:
            self.size = self.labeled_data.shape[0]
            self.labeled_data.reset_index(inplace=True, drop=True)
            self.raw_data = self.raw_data[:self.size]
            self.ta_data = self.ta_data[:self.size]
            self.derive_data = self.derive_data[:self.size]
            self.scaled_data = self.scaled_data[:self.size]
            self.islabeled = True

            self.save(f'./labeled{self.size}.p')

    #sample the data
    def sample(self, features, windowsize):
        """
        returns a list with elements containing:
            -a sample of data
            -a label
            -a order
        """
        data = pd.DataFrame()
        data2 = self.labeled_data.copy()

        #choose the columns
        for feature in features:
            data[feature] = data2[feature]

        #save the shape of data and convert into array
        shape = data.shape
        array = np.array(data)

        #flatten the array
        array = array.flatten()

        features_amount = shape[1]
        #amount of elements in one window
        window_elements = windowsize*features_amount
        #amount of elements to skip in next window
        elements_skip = features_amount
        #amount of windows to generate
        windows_amount = shape[0]-windowsize+1

        #define the indexer
        indexer = np.arange(window_elements)[None, :] + elements_skip*np.arange(windows_amount)[:, None]

        #get the samples
        samples = array[indexer].reshape(windows_amount, windowsize, features_amount)

        #get the labels
        labels_array = np.array(data2['bsh'])
        indexer2 = np.arange(windowsize-1, windowsize+windows_amount-1, step=1)
        labels = labels_array[indexer2]

        sampled_list = []
        for i in range(samples.shape[0]):
            sample = samples[i]
            label = labels[i]
            order = i
            appender = [sample, label, order]
            sampled_list.append(appender)

        return sampled_list

    #split the data
    @staticmethod
    def split(sampled_list, splitpercentage):
        train_data = sampled_list[:int(len(sampled_list)*splitpercentage)]
        test_data= sampled_list[int(len(sampled_list)*splitpercentage):]

        return train_data, test_data
 
    #balance the data
    @staticmethod
    def balance(train_samples):
        #balance the train_samples
        hold = []
        buy = []
        sell = []

        for element in train_samples:
            if element[1] == 0:
                hold.append(element)
            elif element[1] == 1:
                buy.append(element)
            elif element[1] == 2:
                sell.append(element)

        maxindex = min(len(hold), len(buy), len(sell))
        hold = hold[:maxindex]
        buy = buy[:maxindex]
        sell = sell[:maxindex]
        
        samples = hold + buy + sell

        #sort the samples
        samples.sort(key=lambda x: x[2])

        return samples

    #extract the labels and samples
    @staticmethod
    def extract(train_samples, test_samples):
        trainX, trainY, testX, testY = [], [], [], []

        for element in train_samples:
            trainX.append(element[0])
            trainY.append(element[1])
        for element in test_samples:
            testX.append(element[0])
            testY.append(element[1])

        return np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)

    #get the data for training
    def get(self, features, windowsize=60, splitpercentage=0.8, balance_traindata=True):
        """
        Returns four numpy arrays with samples/labels
        """
        if not self.islabeled:
            print('you cant get the data which isnt completely labeled')
            return
        #sample the data
        sampled_list = self.sample(features, windowsize)

        #split the data
        train_data, test_data = self.split(sampled_list, splitpercentage)

        #balance the data
        if balance_traindata == True:
            train_data = self.balance(train_data)

        return self.extract(train_data, test_data)

    #get batches of sampled data
    def getbatches(self, batchsize, features,  windowsize=60, splitpercentage=0.8, balance_traindata=True):
        
        trainX, trainY, testX, testY = self.get(features, windowsize, splitpercentage, balance_traindata)

        #return the batched data
        maxbatchestrain = int(trainX.shape[0] / batchsize)
        trainXc = trainX[0:maxbatchestrain*batchsize]
        trainYc = trainY[0:maxbatchestrain*batchsize]
        maxbatchestest = int(testX.shape[0] / batchsize)
        testXc = testX[0:maxbatchestest*batchsize]
        testYc = testY[0:maxbatchestest*batchsize]

        trainXc = trainXc.reshape((maxbatchestrain, batchsize, windowsize, len(features)))
        trainYc = trainYc.reshape(maxbatchestrain, batchsize)
        testXc = testXc.reshape((maxbatchestest, batchsize, windowsize, len(features)))
        testYc = testYc.reshape(maxbatchestest, batchsize)

        return trainXc, trainYc, testXc, testYc

    #alternativa constructor
    @staticmethod
    def from_database(databasepath):
        obj = pickle.load(open(databasepath, "rb"))
        if isinstance(obj, SupervisedDataBase):
            return obj
        else:
            raise EnvironmentError('You tried to load a file which is not an instance of DataBase')

    #save the object
    def save(self, databasepath):
        pickle.dump(self, open(databasepath, "wb"))


