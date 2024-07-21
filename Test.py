import os
from iqoptionapi.stable_api import IQ_Option
from configparser import ConfigParser
import time
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import BatchNormalization, SimpleRNN, LSTM, GRU, Dense
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import MinMaxScaler
import datetime
import talib as tb
import multiprocessing
from ejtraderIQ import IQOption

import operator
from datetime import datetime, timedelta
import time, json, logging, configparser
from colorama import init, Fore, Back
import sys

dbconf = ConfigParser()
dbconf.read_file(open('config.ini'))
username_IQ = dbconf.get('Config', 'username')
password_IQ = dbconf.get('Config', 'password')
iq_mode = dbconf.get('Config', 'mode')
size = int(dbconf.get('Config', 'size'))
Trade_amount = float(dbconf.get('Config', 'Amount'))
Money = Trade_amount
Trade_martingale = float(dbconf.get('Config', 'Martingale'))
set_TakeProfit = int(dbconf.get('Config', 'TakeProfit'))
set_stoploss = int(dbconf.get('Config', 'Stoploss'))
timeperiod = 30
maxdict = 60
Loop_get_candles = 70

#----------------login----------------
print("---------connect to IqOption!------------")
Iq = IQ_Option(str(username_IQ), str(password_IQ))
Iq.connect()

end_from_time=time.time()
ANS=[]
for i in range(1):
    data=Iq.get_candles("EURUSD", 60, 1000, end_from_time)
    ANS =data+ANS
    end_from_time=int(data[0]["from"])-1
print(ANS)

