from database import SupervisedDataBase, Environment
from logger import ProgressBar
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class VanillaLSTM(nn.Module):

    def __init__(self, features, hidden_size, lstmlayers):
        super().__init__()
        self.features = features
        self.featureslen = len(features)
        self.hidden_size = hidden_size
        self.lstmlayers = lstmlayers

        self.lstm = nn.LSTM(input_size=self.featureslen, hidden_size=hidden_size, batch_first=True, num_layers=lstmlayers, dropout=0.5)
        self.fc = nn.Linear(in_features=self.featureslen, out_features=3)

    def forward(self, t):

        t, (hn, cn) = self.lstm(t)

        #get the last cell output
        t = t[:,-1,:]

        #fc layer with sigmoid activation
        t = self.fc(t)

        return t

    def inspect(self, symbol, pause=0.001, stepper=False, loops=1):
        """
        lets the network run in an environment and plots it
        """
        #setup variables
        done = False

        #setup the environment
        env = Environment(1000, symbol, features=self.features)
        o = env.reset()

        #setup the logger
        logger = pd.DataFrame()
        logger['close'] = env.ta_data['close'][env.episode_index:env.episode_index+env.windowsize].copy().reset_index(drop=True)
        logger['close_copy'] = env.scaled_data['close'][env.episode_index:env.episode_index+env.windowsize].copy().reset_index(drop=True)
        logger['hold'] = np.nan
        logger['buy'] = np.nan
        logger['sell'] = np.nan
        logger.reset_index(drop=True, inplace=True)

        #setup the plotting
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        fig.show()

        #mainloop
        torch.set_grad_enabled(False)
        for i in range(loops):
            while not done:

                #get the prediction from the model
                tensor = torch.tensor(o, dtype=torch.float32)
                tensor = tensor.unsqueeze(dim=0)
                prediction = self.forward(tensor)
                prediction = F.softmax(prediction, dim=1)
                maxindex = torch.argmax(prediction).item()
                
                #update the logger
                if maxindex == 0:
                    logger['hold'].iloc[-1] = logger['close'].iloc[-1]
                elif maxindex == 1:
                    logger['buy'].iloc[-1] = logger['close'].iloc[-1]
                elif maxindex == 2:
                    logger['sell'].iloc[-1] = logger['close'].iloc[-1]

                #plotting
                df = pd.DataFrame(o)
                ax.cla()
                ax2.cla()
                ax.plot(df[0])
                ax.plot(logger['close_copy'], color='red', linestyle='--')
                ax2.plot(logger['close'], color='purple')
                ax2.plot(logger['hold'], color='gray', marker='o')
                ax2.plot(logger['buy'], color='green', marker='o')
                ax2.plot(logger['sell'], color='red', marker='o')
                fig.canvas.draw()
                plt.pause(pause)

                #update the environment
                o, r, done = env.step(1)

                #update the logger
                logger.drop(inplace=True, index=0)
                appender = pd.DataFrame({
                    'close': [env.ta_data['close'][env.episode_index+env.windowsize-1].copy()],
                    'close_copy' : [env.scaled_data['close'][env.episode_index+env.windowsize-1].copy()],
                    'hold': [None],
                    'buy': [None],
                    'sell': [None]
                })
                logger = logger.append(appender, ignore_index=True)
                logger.reset_index(drop=True, inplace=True)

                if stepper:
                    input()

        torch.set_grad_enabled(True)

    def train(self, databasepath, epochs, batchsize, balance=True, learningrate=0.001, terminallogger=True, saving=True):
        """
        trains the network
        """

        #setup the dataset
        data = SupervisedDataBase.from_database(databasepath)

        #setup the optimizer
        optimizer = optim.Adam(self.parameters(), lr=learningrate)
        
        #mainloop
        addloss = 0
        addacc = 0
        for i in range(epochs):

            #get the batches
            trainX, trainY, testX, testY = data.getbatches(batchsize=batchsize, features=self.features, balance_traindata=balance)

            #setup the progressbar
            bar = ProgressBar(f'Epoch {i}', maximum=trainX.shape[0]) if terminallogger else 0

            #train the network
            torch.set_grad_enabled(True)
            addloss = 0.0
            for i, element in enumerate(trainX):
                samples = torch.tensor(element, dtype=torch.float32)
                labels = torch.tensor(trainY[i], dtype=torch.int64)
                predictions = self.forward(samples)
                loss = F.cross_entropy(predictions, labels)
                addloss += loss.item()
                loss.backward()
                optimizer.step()

                if terminallogger:
                    bar.step(addloss/(i+1))

            #evaluate the network
            torch.set_grad_enabled(False)
            addloss = 0.0
            addacc = 0.0
            for i, element in enumerate(testX):
                samples = torch.tensor(element, dtype=torch.float32)
                labels = torch.tensor(testY[i], dtype=torch.int64)
                prediction = self.forward(samples)
                loss = F.cross_entropy(prediction, labels)
                addloss += loss.item()
                addacc += self.calc_accuracy(prediction, labels)

            if terminallogger:
                bar.lastcall(addacc/testX.shape[0], addloss/testX.shape[0])
                bar.finish()

        if saving:
            torch.save(self, f'./models/VanillaLSTM_{self.featureslen}_{self.lstmlayers}_{self.hidden_size}')

        return addloss/testX.shape[0], addacc/testX.shape[0]

    @staticmethod
    def calc_accuracy(predictions, labels):
        predictions = F.softmax(predictions, dim=1)
        maxindex = predictions.argmax(dim=1)

        score = 0
        for i, element in enumerate(maxindex):
            if element == labels[i]:
                score += 1
        
        return score/predictions.shape[0]