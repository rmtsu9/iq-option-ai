import os
from iqoptionapi.stable_api import IQ_Option
from configparser import ConfigParser
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import BatchNormalization, SimpleRNN, LSTM, GRU, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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
from colorama import init, Fore, Back
import time

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
Loop_get_candles = 60
#----------------login----------------
print("---------connect to IqOption!------------")
Iq = IQ_Option(str(username_IQ), str(password_IQ))
Iq.connect()
iq = Iq
BA = Iq.get_balance()
BAty = Iq.get_currency()
if Iq != None:
  while True:
    if Iq.check_connect() == False:
        print('Error when trying to connect')
        print(Iq)
        print("Retrying")
        Iq.connect()
    else:
        time.sleep(1)
        print("--------Successfully Connected!----------")
        print("------------Have a nice day--------------")
        print("-----------", BAty, " - ", BA, "------------")
    break
time.sleep(1)
Iq.change_balance(iq_mode)
print("-----------------------------------------")
print('--------Welcome to AI IQ-Option----------')
print('-Facebook Contact : Supachai Sornprasert-')
print("-----------------------------------------")
#--------------------------------------
taset_finalD_path = 'data/0.Train_data.csv'
Decision_finalD_path = 'data/0.Decision_data.csv'
if os.path.exists(taset_finalD_path):
    print("Delete Old Data")
    os.remove('data/0.Train_data.csv')
    os.remove('data/0000.FulldataAmout.csv')
    if os.path.exists('data/0000.markets.dayly.csv'):
        os.remove('data/0000.markets.dayly.csv')
    if os.path.exists(Decision_finalD_path):
        os.remove('data/0.PATTERN_data.csv')
        os.remove('data/0.Decision_data.csv')

#--------------------------------------
BA = Iq.get_balance()
GetBA = BA
TakeProfit = GetBA + set_TakeProfit
stoploss = GetBA + set_stoploss
#--------------------------------------
market_all_L = 0
#--------------------------------------
while True:
    # ----------------login----------------
    print("---------connect to IqOption!------------")
    Iq = IQ_Option(str(username_IQ), str(password_IQ))
    Iq.connect()
    iq = Iq
    BA = Iq.get_balance()
    BAty = Iq.get_currency()
    if Iq != None:
        while True:
            if Iq.check_connect() == False:
                print('Error when trying to connect')
                print(Iq)
                print("Retrying")
                Iq.connect()
            else:
                print("-----------", BAty, " - ", BA, "------------")
            break
    time.sleep(1)
    Iq.change_balance(iq_mode)
    # # --------------------------------------
    # while True:
    #     print("------------------Date-------------------")
    #     ddtt = datetime.datetime.now()
    #     Timelistrun = ["00", "04", "07"]
    #     Truedaynow = ddtt.strftime("%A")
    #     Timeindaynow = ddtt.strftime("%H")
    #     print(Truedaynow, Timeindaynow, ddtt)
    #     if Timeindaynow in Timelistrun:
    #         Min_sec = 30 * 60
    #         time.sleep(Min_sec)
    #     else:
    #         break
    #     break
    # # --------------------------------------
    print("------------Connected-Market-------------")
    P_open = Iq.get_all_open_time()
    market_O = []
    catalogacao = {}
    for par in P_open['binary']:
        if P_open['binary'][par]['open'] == True:
            timer = int(time.time())
            market_O.append(par)
            print(Fore.GREEN + '*' + Fore.RESET + ' CATALOGING - ' + par + '.. ', end='')
            print('ended in ' + str(int(time.time()) - timer) + ' seconds')
    market_loop = int(len(market_O))
    if market_all_L < market_loop:
        market = market_O[market_all_L]
        goal = str(market)
        asset = str(market)
        ACTIVES = str(market)
        Actives = str(market)
        market_all_L = market_all_L + 1
    else:
        market = market_O[0]
        goal = str(market)
        asset = str(market)
        ACTIVES = str(market)
        Actives = str(market)
        market_all_L = 0
    # --------------------------------------
    print("--------------Loadding IQ----------------")
    BA = Iq.get_balance()
    GetBA = BA
    print('Market =', Actives)
    print('Money =', GetBA)
    print('TakeProfit =',TakeProfit)
    print('Stoploss =', stoploss)
    if TakeProfit == GetBA:
        exit()
    elif stoploss == GetBA:
        exit()

    # Rec time
    start_time = time.time()

    print('LODDING...Candles_DATA...')
    end_from_time = time.time()
    ANS = []
    can_get = 916
    for i in range(Loop_get_candles):
        data = Iq.get_candles(Actives, 60, can_get, end_from_time)
        if data:
            ANS = data + ANS
            end_from_time = int(data[0]["from"]) - 1
        else:
            print(f"No data retrieved in iteration {i}")
    if ANS:
        df = pd.DataFrame(ANS)
        action_df = df[['id', 'from', 'at', 'to']].copy()
        taset_df = df[['open', 'close', 'max', 'min', 'volume']].copy()
        action_df.columns = ['Id', 'From', 'At', 'To']
        taset_df.columns = ['Open', 'Close', 'High', 'Low', 'Volume']
    else:
        print("No data was retrieved.")

    print('LODDING...TECHNICHEN...')
    o = taset_df['Open'].values
    c = taset_df['Close'].values
    h = taset_df['High'].values
    l = taset_df['Low'].values
    v = taset_df['Volume'].astype(float).values
    ta = pd.DataFrame()
    ta["High/Open"] = h / o
    ta["Low/Open"] = l / o
    ta["Close/Open"] = c / o
    ta["AVGPRICE"] = tb.AVGPRICE(o, h, l, c)
    ta["MEDPRICE"] = tb.MEDPRICE(h, l)
    ta["TYPPRICE"] = tb.TYPPRICE(h, l, c)
    ta["WCLPRICE"] = tb.WCLPRICE(h, l, c)
    # MA
    ta['MA5'] = tb.MA(c, timeperiod=5)
    ta['MA14'] = tb.MA(c, timeperiod=14)
    ta['MA16'] = tb.MA(c, timeperiod=16)
    # SMA
    ta['SMA5'] = tb.SMA(c, timeperiod=5)
    ta['SMA14'] = tb.SMA(c, timeperiod=14)
    ta['SMA16'] = tb.SMA(c, timeperiod=16)
    # EMA
    ta['EMA5'] = tb.SMA(c, timeperiod=5)
    ta['EMA14'] = tb.SMA(c, timeperiod=14)
    ta['EMA16'] = tb.SMA(c, timeperiod=16)
    # BBANDS
    ta['BBANDS_U5'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0]
    ta['BBANDS_M5'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1]
    ta['BBANDS_L5'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2]
    # RSI
    ta['RSI14'] = tb.RSI(c, timeperiod=14)
    ta['RSI16'] = tb.RSI(c, timeperiod=16)

    selected_columns = ta.copy()
    taset_df = pd.concat([taset_df[['Open', 'Close', 'High', 'Low', 'Volume']], selected_columns], axis=1)
    taset_final = taset_df.fillna(0)
    new_data = pd.DataFrame(taset_final.copy())
    taset_finalD_path = 'data/0.Train_data.csv'
    if os.path.exists(taset_finalD_path):
        taset_finalD = pd.read_csv(taset_finalD_path)
    else:
        taset_finalD = pd.DataFrame()
    taset_finalD = pd.concat([taset_finalD, new_data], ignore_index=True)
    taset_finalD = taset_finalD.drop(df.index[0:16])
    taset_finalD.to_csv(taset_finalD_path, index=False)
    taset_finalD.to_csv('data/0000.FulldataAmout.csv', index=False)
    taset_row = taset_finalD.shape[0]
    taset_calum = taset_finalD.shape[1]
    taset_values = taset_calum * taset_row

    while True:
        print("--------------Loadding Brain----------------")
        rael_money = Money
        # -----------------------------------------------------------------
        gpus = tf.config.experimental.list_physical_devices('GPU')
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except Exception as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
        # ------------------------------------------------------------------
        databatchsize = "data/0000.FulldataAmout.csv"
        data_batch_size = pd.read_csv(databatchsize)
        data_row = data_batch_size.shape[0]
        data_calum = data_batch_size.shape[1]
        Len_of_batch_size = data_row * data_calum
        print("Now_Data:  ", taset_values)
        print("Full_Data: ", Len_of_batch_size)
        DATA_FILE = "data/0.Train_data.csv"
        data_file = "data/0.Train_data.csv"
        learning_rate = 0.001
        TEST_SIZE = 0.2
        RANDOM_STATE = 42
        BATCH_SIZE = 32
        LEARNING_RATE = learning_rate
        TRAIN_SPLIT = 0.8
        EPOCHS = 100
        NEGATIVE_LOSS_THRESHOLD = -2.0
        # ------------------------------------------------------------------
        # Load the data
        def load_and_preprocess_data(data_file: str, test_size: float, random_state: int) -> tuple:
            data = pd.read_csv(data_file)
            features = data.drop(columns=["Close"])
            target = data["Close"]
            X = features.values
            y = target.values

            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            time_steps = 1
            X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], time_steps, X_train_scaled.shape[1])
            X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], time_steps, X_val_scaled.shape[1])
            X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], time_steps, X_test_scaled.shape[1])

            input_shape_rec = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])

            return X_train_reshaped, X_test_reshaped, y_train, y_test, X_val_reshaped, y_val, scaler, input_shape_rec

        def build_models(input_shape_rec: tuple, learning_rate: float) -> tuple:
            with tf.device('/device:GPU:0'):
                optimizer_Adam = Adam(learning_rate=learning_rate)
                optimizer_Nadam = Nadam(learning_rate=learning_rate)

                rnn_model = Sequential([
                    BatchNormalization(input_shape=input_shape_rec),
                    SimpleRNN(units=128, activation='relu', return_sequences=True, kernel_initializer=glorot_uniform()),
                    SimpleRNN(units=128, activation='relu', return_sequences=False,
                              kernel_initializer=glorot_uniform()),
                    Dense(units=1, activation='linear')
                ])
                rnn_model.compile(optimizer=optimizer_Adam, loss='mean_squared_error', metrics=['mae'])

                lstm_model = Sequential([
                    BatchNormalization(input_shape=input_shape_rec),
                    LSTM(units=128, activation='relu', return_sequences=True, kernel_initializer=glorot_uniform()),
                    LSTM(units=128, activation='relu', return_sequences=False, kernel_initializer=glorot_uniform()),
                    Dense(units=1, activation='linear')
                ])
                lstm_model.compile(optimizer=optimizer_Nadam, loss='mean_squared_error', metrics=['mae'])

            return rnn_model, lstm_model

        class NegativeLossStop(tf.keras.callbacks.Callback):
            def __init__(self, threshold):
                super().__init__()
                self.threshold = threshold

            def on_epoch_end(self, epoch, logs=None):
                current_loss = logs.get('loss')
                if current_loss is not None and current_loss < self.threshold:
                    print(f"\nStopping training as loss ({current_loss}) is below threshold ({self.threshold}).")
                    self.model.stop_training = True

        def main():
            X_train, X_test, y_train, y_test, X_val, y_val, scaler, input_shape_rec = load_and_preprocess_data(
                DATA_FILE, TEST_SIZE, RANDOM_STATE)

            rnn_model, lstm_model = build_models(input_shape_rec, LEARNING_RATE)

            if not os.path.exists("All_model"):
                os.makedirs("All_model")
            if not os.path.exists("All_model/rnn_model"):
                os.makedirs("All_model/rnn_model")
            if not os.path.exists("All_model/lstm_model"):
                os.makedirs("All_model/lstm_model")

            rnn_model.save_weights('All_model/rnn_model/my_rnn_model.h5')
            lstm_model.save_weights('All_model/lstm_model/my_lstm_model.h5')

            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
            negative_loss_callback = NegativeLossStop(threshold=NEGATIVE_LOSS_THRESHOLD)
            checkpoint_path = "All_model/training_1/cp.ckpt"
            checkpoint_dir = os.path.dirname(checkpoint_path)
            ep5_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path.format(epoch=0), verbose=1,
                                                              save_weights_only=True, save_freq=5 * BATCH_SIZE)

            best_model = None
            best_loss = float('inf')

            for model, model_name in zip([lstm_model, rnn_model], ['lstm', 'rnn']):
                with tf.device('/device:GPU:0'):
                    callbacks = [ep5_callback, early_stopping, negative_loss_callback]
                    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
                              verbose=1, callbacks=callbacks)

                test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
                print(f"{model_name.upper()} Test Loss: {test_loss:.5f}, MAE: {test_mae:.5f}")

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_model = model

                model.save(f'All_model/tf_best_model_{model_name}.h5')

            rnn_loss, rnn_mae = rnn_model.evaluate(X_test, y_test, verbose=1)
            lstm_loss, lstm_mae = lstm_model.evaluate(X_test, y_test, verbose=1)

            print(f"RNN Test Loss: {rnn_loss:.5f}, MAE: {rnn_mae:.5f}")
            print(f"LSTM Test Loss: {lstm_loss:.5f}, MAE: {lstm_mae:.5f}")

            rnn_model.save_weights('All_model/rnn_model/my_rnn_model.h5')
            lstm_model.save_weights('All_model/lstm_model/my_lstm_model.h5')


        if __name__ == "__main__":
            main()

        # # -----------------------------------------------------------------------------------------------------------------------------------------------------------
        #
        # end_time = time.time()
        # execution_Candles = end_time - start_time
        # execution_time = execution_Candles/60
        # print("Count time of Train: ", execution_time, " min")
        # print('LODDING Candles: ', execution_time, " min")
        # ANS_get = []
        # can_get = int(execution_Candles/60)
        # can_get = can_get * 60
        # can_get = can_get + 16
        # end_from_time = time.time()
        # for i in range(1):
        #     data = Iq.get_candles(Actives, 60, can_get, end_from_time)
        #     ANS_get = data + ANS_get
        #     end_from_time = int(data[0]["from"]) - 1
        #     d_get = pd.DataFrame(ANS_get)
        #     action_df = d_get[['id', 'from', 'at', 'to']].copy()
        #     taset_df = d_get[['open', 'close', 'max', 'min', 'volume']].copy()
        #     action_df.columns = ['Id', 'From', 'At', 'To']
        #     taset_df.columns = ['Open', 'Close', 'High', 'Low', 'Volume']
        #
        # print('LODDING...TECHNICHEN...')
        # o = taset_df['Open'].values
        # c = taset_df['Close'].values
        # h = taset_df['High'].values
        # l = taset_df['Low'].values
        # v = taset_df['Volume'].astype(float).values
        # ta = pd.DataFrame()
        # ta["High/Open"] = h / o
        # ta["Low/Open"] = l / o
        # ta["Close/Open"] = c / o
        # ta["AVGPRICE"] = tb.AVGPRICE(o, h, l, c)
        # ta["MEDPRICE"] = tb.MEDPRICE(h, l)
        # ta["TYPPRICE"] = tb.TYPPRICE(h, l, c)
        # ta["WCLPRICE"] = tb.WCLPRICE(h, l, c)
        # # MA
        # ta['MA5'] = tb.MA(c, timeperiod=5)
        # ta['MA14'] = tb.MA(c, timeperiod=14)
        # ta['MA16'] = tb.MA(c, timeperiod=16)
        # # SMA
        # ta['SMA5'] = tb.SMA(c, timeperiod=5)
        # ta['SMA14'] = tb.SMA(c, timeperiod=14)
        # ta['SMA16'] = tb.SMA(c, timeperiod=16)
        # # EMA
        # ta['EMA5'] = tb.SMA(c, timeperiod=5)
        # ta['EMA14'] = tb.SMA(c, timeperiod=14)
        # ta['EMA16'] = tb.SMA(c, timeperiod=16)
        # # BBANDS
        # ta['BBANDS_U5'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0]
        # ta['BBANDS_M5'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1]
        # ta['BBANDS_L5'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2]
        # # RSI
        # ta['RSI14'] = tb.RSI(c, timeperiod=14)
        # ta['RSI16'] = tb.RSI(c, timeperiod=16)
        #
        # pcan = pd.DataFrame()
        # open = o
        # close = c
        # high = h
        # low = l
        # volume = v
        # pcan['CDL2CROWS'] = tb.CDL2CROWS(open, h, low, close)
        # pcan['CDL3BLACKCROWS'] = tb.CDL3BLACKCROWS(open, high, low, close)
        # pcan['CDL3INSIDE'] = tb.CDL3INSIDE(open, high, low, close)
        # pcan['CDL3LINESTRIKE'] = tb.CDL3LINESTRIKE(open, high, low, close)
        # pcan['CDL3OUTSIDE'] = tb.CDL3OUTSIDE(open, high, low, close)
        # pcan['CDL3STARSINSOUTH'] = tb.CDL3STARSINSOUTH(open, high, low, close)
        # pcan['CDL3WHITESOLDIERS'] = tb.CDL3WHITESOLDIERS(open, high, low, close)
        # pcan['CDLABANDONEDBABY'] = tb.CDLABANDONEDBABY(open, high, low, close, penetration=0)
        # pcan['CDLADVANCEBLOCK'] = tb.CDLADVANCEBLOCK(open, high, low, close)
        # pcan['CDLBELTHOLD'] = tb.CDLBELTHOLD(open, high, low, close)
        # pcan['CDLBREAKAWAY'] = tb.CDLBREAKAWAY(open, high, low, close)
        # pcan['CDLCLOSINGMARUBOZU'] = tb.CDLCLOSINGMARUBOZU(open, high, low, close)
        # pcan['CDLCONCEALBABYSWALL'] = tb.CDLCONCEALBABYSWALL(open, high, low, close)
        # pcan['CDLCOUNTERATTACK'] = tb.CDLCOUNTERATTACK(open, high, low, close)
        # pcan['CDLDARKCLOUDCOVER'] = tb.CDLDARKCLOUDCOVER(open, high, low, close, penetration=0)
        # pcan['CDLDOJI'] = tb.CDLDOJI(open, high, low, close)
        # pcan['CDLDOJISTAR'] = tb.CDLDOJISTAR(open, high, low, close)
        # pcan['CDLDRAGONFLYDOJI'] = tb.CDLDRAGONFLYDOJI(open, high, low, close)
        # pcan['CDLENGULFING'] = tb.CDLENGULFING(open, high, low, close)
        # pcan['CDLEVENINGDOJISTAR'] = tb.CDLEVENINGDOJISTAR(open, high, low, close, penetration=0)
        # pcan['CDLEVENINGSTAR'] = tb.CDLEVENINGSTAR(open, high, low, close, penetration=0)
        # pcan['CDLGAPSIDESIDEWHITE'] = tb.CDLGAPSIDESIDEWHITE(open, high, low, close)
        # pcan['CDLGRAVESTONEDOJI'] = tb.CDLGRAVESTONEDOJI(open, high, low, close)
        # pcan['CDLHAMMER'] = tb.CDLHAMMER(open, high, low, close)
        # pcan['CDLHANGINGMAN'] = tb.CDLHANGINGMAN(open, high, low, close)
        # pcan['CDLHARAMI'] = tb.CDLHARAMI(open, high, low, close)
        # pcan['CDLHARAMICROSS'] = tb.CDLHARAMICROSS(open, high, low, close)
        # pcan['CDLHIGHWAVE'] = tb.CDLHIGHWAVE(open, high, low, close)
        # pcan['CDLHIKKAKE'] = tb.CDLHIKKAKE(open, high, low, close)
        # pcan['CDLHIKKAKEMOD'] = tb.CDLHIKKAKEMOD(open, high, low, close)
        # pcan['CDLHOMINGPIGEON'] = tb.CDLHOMINGPIGEON(open, high, low, close)
        # pcan['CDLIDENTICAL3CROWS'] = tb.CDLIDENTICAL3CROWS(open, high, low, close)
        # pcan['CDLINNECK'] = tb.CDLINNECK(open, high, low, close)
        # pcan['CDLINVERTEDHAMMER'] = tb.CDLINVERTEDHAMMER(open, high, low, close)
        # pcan['CDLKICKING'] = tb.CDLKICKING(open, high, low, close)
        # pcan['CDLKICKINGBYLENGTH'] = tb.CDLKICKINGBYLENGTH(open, high, low, close)
        # pcan['CDLLADDERBOTTOM'] = tb.CDLLADDERBOTTOM(open, high, low, close)
        # pcan['CDLLONGLEGGEDDOJI'] = tb.CDLLONGLEGGEDDOJI(open, high, low, close)
        # pcan['CDLLONGLINE'] = tb.CDLLONGLINE(open, high, low, close)
        # pcan['CDLMARUBOZU'] = tb.CDLMARUBOZU(open, high, low, close)
        # pcan['CDLMATCHINGLOW'] = tb.CDLMATCHINGLOW(open, high, low, close)
        # pcan['CDLMATHOLD'] = tb.CDLMATHOLD(open, high, low, close, penetration=0)
        # pcan['CDLMORNINGDOJISTAR'] = tb.CDLMORNINGDOJISTAR(open, high, low, close, penetration=0)
        # pcan['CDLMORNINGSTAR'] = tb.CDLMORNINGSTAR(open, high, low, close, penetration=0)
        # pcan['CDLONNECK'] = tb.CDLONNECK(open, high, low, close)
        # pcan['CDLPIERCING'] = tb.CDLPIERCING(open, high, low, close)
        # pcan['CDLRICKSHAWMAN'] = tb.CDLRICKSHAWMAN(open, high, low, close)
        # pcan['CDLRISEFALL3METHODS'] = tb.CDLRISEFALL3METHODS(open, high, low, close)
        # pcan['CDLSEPARATINGLINES'] = tb.CDLSEPARATINGLINES(open, high, low, close)
        # pcan['CDLSHOOTINGSTAR'] = tb.CDLSHOOTINGSTAR(open, high, low, close)
        # pcan['CDLSHORTLINE'] = tb.CDLSHORTLINE(open, high, low, close)
        # pcan['CDLSPINNINGTOP'] = tb.CDLSPINNINGTOP(open, high, low, close)
        # pcan['CDLSTALLEDPATTERN'] = tb.CDLSTALLEDPATTERN(open, high, low, close)
        # pcan['CDLSTICKSANDWICH'] = tb.CDLSTICKSANDWICH(open, high, low, close)
        # pcan['CDLTAKURI'] = tb.CDLTAKURI(open, high, low, close)
        # pcan['CDLTASUKIGAP'] = tb.CDLTASUKIGAP(open, high, low, close)
        # pcan['CDLTHRUSTING'] = tb.CDLTHRUSTING(open, high, low, close)
        # pcan['CDLTRISTAR'] = tb.CDLTRISTAR(open, high, low, close)
        # pcan['CDLUNIQUE3RIVER'] = tb.CDLUNIQUE3RIVER(open, high, low, close)
        # pcan['CDLUPSIDEGAP2CROWS'] = tb.CDLUPSIDEGAP2CROWS(open, high, low, close)
        # pcan['CDLXSIDEGAP3METHODS'] = tb.CDLXSIDEGAP3METHODS(open, high, low, close)
        # pcan.to_csv('data/0.PATTERN_data.csv', index=False)
        #
        # selected_columns = ta.copy()
        # taset_df = pd.concat([taset_df[['Open', 'Close', 'High', 'Low', 'Volume']], selected_columns], axis=1)
        # taset_final = taset_df.fillna(0)
        # Decision_data = pd.DataFrame(taset_final.copy())
        # Decision_finalD_path = 'data/0.Decision_data.csv'
        # if os.path.exists(Decision_finalD_path):
        #     Decision_finalD = pd.read_csv(Decision_finalD_path)
        # else:
        #     Decision_finalD = pd.DataFrame()
        # Decision_finalD = pd.concat([Decision_finalD, Decision_data], ignore_index=True)
        # Decision_finalD = Decision_finalD.drop(d_get.index[0:16])
        # Decision_finalD.to_csv(Decision_finalD_path, index=False)
        #
        # # Decision system
        # def make_predictions(model, X_test):
        #     predictions = model.predict(X_test)
        #     return predictions
        # def make_trading_decision(predictions, threshold):
        #     decisions = []
        #     global rael_money, Actives, Money
        #
        #     PATTERN_file = 'data/0.PATTERN_data.csv'
        #     pattern_data = pd.read_csv(PATTERN_file)
        #     last_row = pattern_data.iloc[-1]
        #     count_negative_100 = (last_row <= -100).sum()
        #     count_zero = (last_row == 0).sum()
        #     count_positive_100 = (last_row >= 100).sum()
        #     print("PATTERN Candles")
        #     print("PATTERN -100 : ", count_negative_100)
        #     print("PATTERN 0 : ", count_zero)
        #     print("PATTERN 100 : ", count_positive_100)
        #
        #     count_negative_100 = count_negative_100 * -1
        #     count_positive_100 = count_positive_100 * 1
        #     count_ans = count_positive_100 + count_negative_100
        #     print("PATTERN ANS : ", count_ans)
        #     if count_ans < -2:
        #         threshold = 0.8
        #     elif count_ans > 2:
        #         threshold = 0.4
        #     else:
        #         threshold = threshold
        #
        #     for pred in predictions:
        #         if pred > threshold:
        #             time.sleep(5)
        #             decisions.append('Buy')
        #             print("IQOPTION : BUY")
        #             done, id = iq.buy(rael_money, Actives, "call", 1)  # Buy order iq
        #             if not done:
        #                 print('Error call')
        #                 print(done, id)
        #                 break
        #         elif pred < threshold:
        #             time.sleep(5)
        #             decisions.append('Sell')
        #             done, id = iq.buy(rael_money, Actives, "put", 1)  # Sell order
        #             if not done:
        #                 print('Error put')
        #                 print(done, id)
        #                 break
        #         else:
        #             threshold = 0.5
        #             if pred > threshold:
        #                 time.sleep(5)
        #                 decisions.append('Buy')
        #                 print("IQOPTION : BUY")
        #                 done, id = iq.buy(rael_money, Actives, "call", 1)  # Buy order iq
        #                 if not done:
        #                     print('Error call')
        #                     print(done, id)
        #                     break
        #             elif pred < threshold:
        #                 time.sleep(5)
        #                 decisions.append('Sell')
        #                 done, id = iq.buy(rael_money, Actives, "put", 1)  # Sell order
        #                 if not done:
        #                     print('Error put')
        #                     print(done, id)
        #                     break
        #         time.sleep(80)
        #         print("id : " + str(id))
        #         w_r = iq.get_optioninfo(1)
        #         w_r = w_r['msg']['result']['closed_options'][0]['win']
        #         if w_r == 'win':
        #             win_result = 1
        #             if rael_money > 1.3:
        #                 rael_money = 1.0
        #                 break
        #             else:
        #                 rael_money = rael_money * Trade_martingale
        #             print('Balance : ', BA, " ", BAty)
        #             print("Prediction Result : -- Win --")
        #             print("-----------------------------------------")
        #         else:
        #             win_result = 0
        #             rael_money = Money
        #             print('Balance : ', BA, " ", BAty)
        #             print("Prediction Result : -- Loose --")
        #             print("-----------------------------------------")
        #             break
        #     return decisions
        #
        # Decision_data_file = "data/0.Decision_data.csv"
        # X_train, X_test, y_train, y_test, X_val, y_val, scaler, input_shape_rec = load_and_preprocess_data(data_file, 0.2, 42)
        #
        # threshold = 0.5
        # predictions = make_predictions(lstm_model, X_test)
        # print("-----------------------------------------")
        # print('Bot Trading : ' + str(market))
        # probability_put = int(predictions[0][0] * 100)
        # print('probability of PUT: ' + str(probability_put) + ' %')
        # probability_call = probability_put - 100
        # print('probability of CALL: ' + str(probability_call) + ' %')
        # print("-----------------------------------------")
        # trading_decisions = make_trading_decision(predictions, threshold)
        # print('Trading_decisions : ', trading_decisions)
        # print("-----------------------------------------")
        # time.sleep(10)
        # os.remove('data/0.PATTERN_data.csv')
        # os.remove('data/0.Decision_data.csv')
        os.remove('data/0.Train_data.csv')

        break