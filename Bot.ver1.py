SETup - Copy.pyimport os
from iqoptionapi.stable_api import IQ_Option
from configparser import ConfigParser
import time
import talib as tb
import torch.optim as optim
import multiprocessing
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.layers import SimpleRNN, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import StandardScaler
import datetime

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
learning_rate = float(dbconf.get('Config', 'learning_rate'))
batch_size = int(dbconf.get('Config', 'Batchsize'))
timeperiod = 30
maxdict = 60

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
        print("------", IQ_Option.__version__, "------")
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
taset_finalD_path = "taset_final.csv"
if os.path.exists(taset_finalD_path):
    os.remove("taset_final.csv")
#--------------------------------------
BA = Iq.get_balance()
GetBA = BA
TakeProfit = GetBA + set_TakeProfit
stoploss = GetBA + set_stoploss
#--------------------------------------
while True:
    # --------------------------------------
    while True:
        ddtt = datetime.datetime.now()
        Daylistrun = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        Timelistrun = ["23", "00", "01", "02", "03", "04", "05", "06", "07", "08"]
        Truedaynow = ddtt.strftime("%A")
        Timeindaynow = ddtt.strftime("%H")
        Timeinlistrun = len(Timelistrun)
        Dayinlistrun = len(Daylistrun)
        time.sleep(1)
        for Dayin in range(Dayinlistrun):
            if Truedaynow == Daylistrun[Dayin]:
                for Timein in range(Timeinlistrun):
                    if Timeindaynow != Timelistrun[Timein]:
                        print(Daylistrun[Dayin], Timeindaynow)
                        break

                    elif Timeindaynow == Timelistrun[Timein]:
                        print(Daylistrun[Dayin], Timeindaynow)
                        Minnsecc = 30 * 60
                        print("Stop Run : ", Minnsecc / 60)
                        time.sleep(Minnsecc)
        break
    # --------------------------------------
    market = ['EURUSD', 'EURJPY', 'EURGBP']
    goal = dbconf.get('Config', 'Market')
    asset = dbconf.get('Config', 'Market')
    market = dbconf.get('Config', 'Market')
    ACTIVES = dbconf.get('Config', 'Market')
    Actives = dbconf.get('Config', 'Market')

    BA = Iq.get_balance()
    GetBA = BA
    print('Money =', GetBA)
    print('TakeProfit =',TakeProfit)
    print('Stoploss =', stoploss)
    if TakeProfit == GetBA:
        exit()
    elif stoploss == GetBA:
        exit()

    print('LODDING...Candles_DATA...')
    end_from_time = time.time()
    ANS = []
    can_get = 916
    for i in range(70):
        data = Iq.get_candles(ACTIVES, 60, can_get, end_from_time)
        ANS = data + ANS
        end_from_time = int(data[0]["from"]) - 1
        df = pd.DataFrame(ANS)
        # สร้าง DataFrame ใหม่ (Taset) โดยเลือกเฉพาะคอลัมน์ที่ต้องการจาก DataFrame เดิม (df)
        action_df = df[['id', 'from', 'at', 'to']].copy()
        taset_df = df[['open', 'close', 'max', 'min', 'volume']].copy()
        # เปลี่ยนชื่อคอลัมน์ใน DataFrame ใหม่
        action_df.columns = ['Id', 'From', 'At', 'To']
        taset_df.columns = ['Open', 'Close', 'High', 'Low', 'Volume']

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
    pcan = pd.DataFrame()
    open = o
    close = c
    high = h
    low = l
    volume = v
    pcan['CDL2CROWS'] = tb.CDL2CROWS(open, h, low, close)
    pcan['CDL3BLACKCROWS'] = tb.CDL3BLACKCROWS(open, high, low, close)
    pcan['CDL3INSIDE'] = tb.CDL3INSIDE(open, high, low, close)
    pcan['CDL3LINESTRIKE'] = tb.CDL3LINESTRIKE(open, high, low, close)
    pcan['CDL3OUTSIDE'] = tb.CDL3OUTSIDE(open, high, low, close)
    pcan['CDL3STARSINSOUTH'] = tb.CDL3STARSINSOUTH(open, high, low, close)
    pcan['CDL3WHITESOLDIERS'] = tb.CDL3WHITESOLDIERS(open, high, low, close)
    pcan['CDLABANDONEDBABY'] = tb.CDLABANDONEDBABY(open, high, low, close, penetration=0)
    pcan['CDLADVANCEBLOCK'] = tb.CDLADVANCEBLOCK(open, high, low, close)
    pcan['CDLBELTHOLD'] = tb.CDLBELTHOLD(open, high, low, close)
    pcan['CDLBREAKAWAY'] = tb.CDLBREAKAWAY(open, high, low, close)
    pcan['CDLCLOSINGMARUBOZU'] = tb.CDLCLOSINGMARUBOZU(open, high, low, close)
    pcan['CDLCONCEALBABYSWALL'] = tb.CDLCONCEALBABYSWALL(open, high, low, close)
    pcan['CDLCOUNTERATTACK'] = tb.CDLCOUNTERATTACK(open, high, low, close)
    pcan['CDLDARKCLOUDCOVER'] = tb.CDLDARKCLOUDCOVER(open, high, low, close, penetration=0)
    pcan['CDLDOJI'] = tb.CDLDOJI(open, high, low, close)
    pcan['CDLDOJISTAR'] = tb.CDLDOJISTAR(open, high, low, close)
    pcan['CDLDRAGONFLYDOJI'] = tb.CDLDRAGONFLYDOJI(open, high, low, close)
    pcan['CDLENGULFING'] = tb.CDLENGULFING(open, high, low, close)
    pcan['CDLEVENINGDOJISTAR'] = tb.CDLEVENINGDOJISTAR(open, high, low, close, penetration=0)
    pcan['CDLEVENINGSTAR'] = tb.CDLEVENINGSTAR(open, high, low, close, penetration=0)
    pcan['CDLGAPSIDESIDEWHITE'] = tb.CDLGAPSIDESIDEWHITE(open, high, low, close)
    pcan['CDLGRAVESTONEDOJI'] = tb.CDLGRAVESTONEDOJI(open, high, low, close)
    pcan['CDLHAMMER'] = tb.CDLHAMMER(open, high, low, close)
    pcan['CDLHANGINGMAN'] = tb.CDLHANGINGMAN(open, high, low, close)
    pcan['CDLHARAMI'] = tb.CDLHARAMI(open, high, low, close)
    pcan['CDLHARAMICROSS'] = tb.CDLHARAMICROSS(open, high, low, close)
    pcan['CDLHIGHWAVE'] = tb.CDLHIGHWAVE(open, high, low, close)
    pcan['CDLHIKKAKE'] = tb.CDLHIKKAKE(open, high, low, close)
    pcan['CDLHIKKAKEMOD'] = tb.CDLHIKKAKEMOD(open, high, low, close)
    pcan['CDLHOMINGPIGEON'] = tb.CDLHOMINGPIGEON(open, high, low, close)
    pcan['CDLIDENTICAL3CROWS'] = tb.CDLIDENTICAL3CROWS(open, high, low, close)
    pcan['CDLINNECK'] = tb.CDLINNECK(open, high, low, close)
    pcan['CDLINVERTEDHAMMER'] = tb.CDLINVERTEDHAMMER(open, high, low, close)
    pcan['CDLKICKING'] = tb.CDLKICKING(open, high, low, close)
    pcan['CDLKICKINGBYLENGTH'] = tb.CDLKICKINGBYLENGTH(open, high, low, close)
    pcan['CDLLADDERBOTTOM'] = tb.CDLLADDERBOTTOM(open, high, low, close)
    pcan['CDLLONGLEGGEDDOJI'] = tb.CDLLONGLEGGEDDOJI(open, high, low, close)
    pcan['CDLLONGLINE'] = tb.CDLLONGLINE(open, high, low, close)
    pcan['CDLMARUBOZU'] = tb.CDLMARUBOZU(open, high, low, close)
    pcan['CDLMATCHINGLOW'] = tb.CDLMATCHINGLOW(open, high, low, close)
    pcan['CDLMATHOLD'] = tb.CDLMATHOLD(open, high, low, close, penetration=0)
    pcan['CDLMORNINGDOJISTAR'] = tb.CDLMORNINGDOJISTAR(open, high, low, close, penetration=0)
    pcan['CDLMORNINGSTAR'] = tb.CDLMORNINGSTAR(open, high, low, close, penetration=0)
    pcan['CDLONNECK'] = tb.CDLONNECK(open, high, low, close)
    pcan['CDLPIERCING'] = tb.CDLPIERCING(open, high, low, close)
    pcan['CDLRICKSHAWMAN'] = tb.CDLRICKSHAWMAN(open, high, low, close)
    pcan['CDLRISEFALL3METHODS'] = tb.CDLRISEFALL3METHODS(open, high, low, close)
    pcan['CDLSEPARATINGLINES'] = tb.CDLSEPARATINGLINES(open, high, low, close)
    pcan['CDLSHOOTINGSTAR'] = tb.CDLSHOOTINGSTAR(open, high, low, close)
    pcan['CDLSHORTLINE'] = tb.CDLSHORTLINE(open, high, low, close)
    pcan['CDLSPINNINGTOP'] = tb.CDLSPINNINGTOP(open, high, low, close)
    pcan['CDLSTALLEDPATTERN'] = tb.CDLSTALLEDPATTERN(open, high, low, close)
    pcan['CDLSTICKSANDWICH'] = tb.CDLSTICKSANDWICH(open, high, low, close)
    pcan['CDLTAKURI'] = tb.CDLTAKURI(open, high, low, close)
    pcan['CDLTASUKIGAP'] = tb.CDLTASUKIGAP(open, high, low, close)
    pcan['CDLTHRUSTING'] = tb.CDLTHRUSTING(open, high, low, close)
    pcan['CDLTRISTAR'] = tb.CDLTRISTAR(open, high, low, close)
    pcan['CDLUNIQUE3RIVER'] = tb.CDLUNIQUE3RIVER(open, high, low, close)
    pcan['CDLUPSIDEGAP2CROWS'] = tb.CDLUPSIDEGAP2CROWS(open, high, low, close)
    pcan['CDLXSIDEGAP3METHODS'] = tb.CDLXSIDEGAP3METHODS(open, high, low, close)
    pcan.to_csv("PATTERN_final.csv", index=False)

    selected_columns = ta.copy()
    taset_df = pd.concat([taset_df[['Open', 'Close', 'High', 'Low', 'Volume']], selected_columns], axis=1)
    taset_final = taset_df.fillna(0)
    new_data = pd.DataFrame(taset_final.copy())
    taset_finalD_path = "taset_final.csv"
    if os.path.exists(taset_finalD_path):
        taset_finalD = pd.read_csv(taset_finalD_path)
    else:
        taset_finalD = pd.DataFrame()
    taset_finalD = pd.concat([taset_finalD, new_data], ignore_index=True)
    taset_finalD = taset_finalD.drop(df.index[0:16])
    taset_finalD.to_csv(taset_finalD_path, index=False)

    while True:
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

        os.environ.pop('TF_CONFIG', None)
        print('Number of cpu: ' + str(multiprocessing.cpu_count()))

        # Create a MirroredStrategy.
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        num_cpus = len(tf.config.experimental.list_physical_devices('CPU'))
        print("Number of available CPUs:", num_cpus)

        tf.keras.backend.clear_session()
        memory_log_file = "memory_log.csv"
        if os.path.exists(memory_log_file):
            memory_log = pd.read_csv(memory_log_file)
        else:
            memory_log = pd.DataFrame(columns=[
                "Market", "Id", "From", "At", "To", "Action", "Profit/Loss", "Win", "EnsemblePrediction",
                "LSTM", 'GRU', "RNN", "CNN", "WinCount", "LoseCount", "AI_Focus_Win_Rate"
            ])
            memory_log.to_csv("memory_log.csv", index=False)

        memory_log_data = "memory_log_data_back.csv"
        if os.path.exists(memory_log_data):
            os.remove(memory_log_data)
            memory_log_data_back = pd.DataFrame(columns=[
                "Market", "Id", "From", "At", "To", "Action", "Profit/Loss", "Win", "EnsemblePrediction",
                "LSTM", 'GRU', "RNN", "CNN", "WinCount", "LoseCount", "AI_Focus_Win_Rate"
            ])
            memory_log_data_back.to_csv("memory_log_data_back.csv", index=False)
        else:
            memory_log_data_back = pd.DataFrame(columns=[
                "Market", "Id", "From", "At", "To", "Action", "Profit/Loss", "Win", "EnsemblePrediction",
                "LSTM", 'GRU', "RNN", "CNN", "WinCount", "LoseCount", "AI_Focus_Win_Rate"
            ])
            memory_log_data_back.to_csv("memory_log_data_back.csv", index=False)

        print("Load Features and target data......")
        taset_final_FFFF = "taset_final.csv"
        data = pd.read_csv(taset_final_FFFF)
        features = data.drop(columns=["Close"])
        target = data["Close"]

        tf_X = features.values
        tf_y = target.values
        tf_X_train, tf_X_test, tf_y_train, tf_y_test = train_test_split(tf_X, tf_y, test_size=0.2, random_state=26)


        scaler = StandardScaler()
        tf_X_train = scaler.fit_transform(tf_X_train)
        tf_X_test = scaler.transform(tf_X_test)
        tf_X_train = tf_X_train.reshape((tf_X_train.shape[0], 1, tf_X_train.shape[1]))
        tf_X_test = tf_X_test.reshape((tf_X_test.shape[0], 1, tf_X_test.shape[1]))

        input_shape_rec = (1, tf_X_train.shape[2])
        input_shape_cnn = (tf_X_train.shape[1], 1)
        print("Deep TensorFlow......")
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            optimizer_Adam = Adam(learning_rate=learning_rate)
            optimizer_Nadam = Nadam(learning_rate=learning_rate)
            lstm_file_name = "tf_best_model.h5"
            if os.path.exists(lstm_file_name):
                print("Load models TensorFlow......")
                lstm_model = tf.keras.models.load_model("tf_lstm_model.h5")
                gru_model = tf.keras.models.load_model("tf_gru_model.h5")
                rnn_model = tf.keras.models.load_model("tf_rnn_model.h5")
                cnn_model = tf.keras.models.load_model("tf_cnn_model.h5")
                best_model = tf.keras.models.load_model("tf_best_model.h5")
                lstm_model.load_weights('./lstm_model/my_lstm_model')
                gru_model.load_weights('./gru_model/my_gru_model')
                rnn_model.load_weights('./rnn_model/my_rnn_model')
                cnn_model.load_weights('./cnn_model/my_cnn_model')
            else:
                print("Create models TensorFlow......")
                rnn_model = Sequential()
                rnn_model.add(BatchNormalization(input_shape=input_shape_rec))
                rnn_model.add(SimpleRNN(units=128, activation='relu'))
                rnn_model.add(Dropout(0.2))
                rnn_model.add(Dense(units=1))

                cnn_model = Sequential()
                cnn_model.add(Conv1D(filters=64, kernel_size=3, input_shape=input_shape_rec, padding='same'))
                cnn_model.add(BatchNormalization())
                cnn_model.add(Activation('relu'))
                cnn_model.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
                cnn_model.add(Flatten())
                cnn_model.add(Dense(units=128))
                cnn_model.add(BatchNormalization())
                cnn_model.add(Activation('relu'))
                cnn_model.add(Dense(units=1))

                lstm_model = Sequential()
                lstm_model.add(BatchNormalization(input_shape=input_shape_rec))
                lstm_model.add(LSTM(units=128, activation='relu'))
                lstm_model.add(Dropout(0.2))
                lstm_model.add(Dense(units=1))

                gru_model = Sequential()
                gru_model.add(BatchNormalization(input_shape=input_shape_rec))
                gru_model.add(GRU(units=128, activation='relu'))
                gru_model.add(Dropout(0.2))
                gru_model.add(Dense(units=1))

                lstm_model.compile(optimizer=optimizer_Nadam, loss='binary_crossentropy', metrics=['accuracy'])
                gru_model.compile(optimizer=optimizer_Nadam, loss='binary_crossentropy', metrics=['accuracy'])
                rnn_model.compile(optimizer=optimizer_Adam, loss='binary_crossentropy', metrics=['accuracy'])
                cnn_model.compile(optimizer=optimizer_Adam, loss='binary_crossentropy', metrics=['accuracy'])
                # Save models
                lstm_model.save_weights('./lstm_model/my_lstm_model')
                gru_model.save_weights('./gru_model/my_gru_model')
                rnn_model.save_weights('./rnn_model/my_rnn_model')
                cnn_model.save_weights('./cnn_model/my_cnn_model')
                lstm_model.save("tf_lstm_model.h5")
                gru_model.save("tf_gru_model.h5")
                rnn_model.save("tf_rnn_model.h5")
                cnn_model.save("tf_cnn_model.h5")

        with strategy.scope():

            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            # Train
            batch_size = batch_size
            epochs = int(batch_size * 5)
            validation_data = (tf_X_test, tf_y_test)
            steps_per_epoch = len(tf_X_train) // batch_size
            def lr_schedule(epoch):
                initial_lr = learning_rate
                return initial_lr * 0.9 ** epoch

            class NegativeLossStop(Callback):
                def __init__(self, threshold):
                    super(NegativeLossStop, self).__init__()
                    self.threshold = threshold
                def on_epoch_end(self, epoch, logs=None):
                    current_loss = logs.get('loss')
                    if current_loss is not None and current_loss < self.threshold:
                        print(f"\nStopping training as loss ({current_loss}) is below threshold ({self.threshold}).")
                        self.model.stop_training = True

            early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)
            negative_loss_threshold = -1.0
            negative_loss_callback = NegativeLossStop(threshold=negative_loss_threshold)
            model_checkpoint = ModelCheckpoint(filepath='tf_best_model.h5', monitor='val_loss', save_best_only=True)
            learning_rate_scheduler = LearningRateScheduler(lr_schedule)
            checkpoint_path = "training_1/cp.ckpt"
            checkpoint_dir = os.path.dirname(checkpoint_path)
            ep5_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq=5*batch_size)
            lstm_model.save_weights(checkpoint_path.format(epoch=0))
            gru_model.save_weights(checkpoint_path.format(epoch=0))
            rnn_model.save_weights(checkpoint_path.format(epoch=0))
            cnn_model.save_weights(checkpoint_path.format(epoch=0))
            train_data = tf.data.Dataset.from_tensor_slices((tf_X_train, tf_y_train)).shuffle(98000).batch(batch_size)
            val_data = tf.data.Dataset.from_tensor_slices((tf_X_test, tf_y_test)).batch(batch_size)

            best_file_name = "tf_best_model.h5"
            if os.path.exists(best_file_name):
                best_model = tf.keras.models.load_model("tf_best_model.h5")
                best_model.compile(optimizer=optimizer_Adam, loss='binary_crossentropy', metrics=['accuracy'])
                best_model.fit(tf_X_train, tf_y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=1, steps_per_epoch=steps_per_epoch,
                               callbacks=[ep5_callback, model_checkpoint, learning_rate_scheduler, negative_loss_callback], use_multiprocessing=True)
                best_model.save("tf_best_model.h5")

            print("lstm_model..........")
            lstm_model.fit(tf_X_train, tf_y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=1, steps_per_epoch=steps_per_epoch,
                             callbacks=[ep5_callback, model_checkpoint, learning_rate_scheduler, negative_loss_callback], use_multiprocessing=True)

            print("gru_model..........")
            gru_model.fit(tf_X_train, tf_y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=1, steps_per_epoch=steps_per_epoch,
                           callbacks=[ep5_callback, model_checkpoint, learning_rate_scheduler, negative_loss_callback], use_multiprocessing=True)

            print("rnn_model..........")
            rnn_model.fit(tf_X_train, tf_y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=1, steps_per_epoch=steps_per_epoch,
                           callbacks=[ep5_callback, model_checkpoint, learning_rate_scheduler, negative_loss_callback], use_multiprocessing=True)

            print("cnn_model..........")
            cnn_model.fit(tf_X_train, tf_y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=1, steps_per_epoch=steps_per_epoch,
                           callbacks=[ep5_callback, model_checkpoint, learning_rate_scheduler, negative_loss_callback], use_multiprocessing=True)


            lstm_model.save("tf_lstm_model.h5")
            gru_model.save("tf_gru_model.h5")
            rnn_model.save("tf_rnn_model.h5")
            cnn_model.save("tf_cnn_model.h5")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------
        # Trading Loop
        rael_money = Money
        for i in range(3):
            print('Running AI Brain Decision')
            # Predictions from all models
            lstm_predictions = lstm_model.predict(tf_X_test)
            gru_predictions = gru_model.predict(tf_X_test)
            rnn_predictions = rnn_model.predict(tf_X_test)
            cnn_predictions = cnn_model.predict(tf_X_test)

            threshold = 0.01
            lstm_direction = 1 if np.any(lstm_predictions > threshold) else -1
            gru_direction = 1 if np.any(gru_predictions > threshold) else -1
            rnn_direction = 1 if np.any(rnn_predictions > threshold) else -1
            cnn_direction = 1 if np.any(cnn_predictions > threshold) else -1


            ensemble_prediction = (lstm_predictions + gru_predictions + rnn_predictions + cnn_predictions) / 4.0
            total_direction = lstm_direction + gru_direction + rnn_direction + cnn_direction
            action1 = 1 if total_direction > 0 else 0
            action2 = 1 if ensemble_prediction > 0 else 0
            action = action1 + action2

            # Execute Trade
            if action == 1:
                print("IQOPTION : BUY")
                done, id = iq.buy(rael_money, Actives, "call", 1)  # Buy order iq
                if not done:
                    print('Error call')
                    print(done, id)
                    break
            else:
                print("IQOPTION : Sell")
                done, id = iq.buy(rael_money, Actives, "put", 1)  # Sell order
                if not done:
                    print('Error put')
                    print(done, id)
                    break
            time.sleep(61)
            print("id : " + str(id))
            w_r = iq.get_optioninfo(1)
            w_r = w_r['msg']['result']['closed_options'][0]['win']
            if w_r == 'win':
                win_result = 1
                if rael_money >= 3:
                    rael_money = 1.0
                rael_money = rael_money * Trade_martingale
                print('Balance : ', BA, " ", BAty)
                print("Prediction Result : -- Win --")
                print("-----------------------------------------")
            else:
                win_result = 0
                rael_money = Money
                print('Balance : ', BA, " ", BAty)
                print("Prediction Result : -- Loose --")
                print("-----------------------------------------")

            memory_log['Win'] = pd.to_numeric(memory_log['Win'], errors='coerce')
            total_trades = i + 1
            total_wins = memory_log['Win'].sum(skipna=True)
            ai_focus_win_rate = total_wins / total_trades if total_trades > 0 else 0

            time.sleep(1)
            print("------------memory_log update------------")
            if i > 0 and 'Close' in taset_final.columns:
                profit_loss = (taset_final.loc[i, 'Close'] - taset_final.loc[i - 1, 'Close']) * action
            else:
                profit_loss = 0
            new_row = {
                "Market": goal,
                'Id': action_df['Id'].mean(),
                'From': action_df['From'].mean(),
                'At': action_df['At'].mean(),
                'To': action_df['To'].mean(),
                'Action': action,
                'Profit/Loss': profit_loss,
                'Win': win_result,
                'EnsemblePrediction': ensemble_prediction.mean(),
                'LSTM': lstm_direction,
                'GRU': gru_direction,
                'RNN': rnn_direction,
                'CNN': cnn_direction,
                'WinCount': memory_log['Win'].sum(),
                'LoseCount': len(memory_log) - memory_log['Win'].sum(),
                'AI_Focus_Win_Rate': ai_focus_win_rate
            }
            memory_log = pd.concat([memory_log, pd.DataFrame([new_row])], ignore_index=True)
            memory_log_data_back = pd.concat([memory_log_data_back, pd.DataFrame([new_row])], ignore_index=True)
            memory_log.to_csv(memory_log_file, index=False)
            memory_log_data_back.to_csv(memory_log_data, index=False)
            time.sleep(1)
            reward = 1.0 if win_result == 1 else -1.0

            lstm_model.save("tf_lstm_model.h5")
            gru_model.save("tf_gru_model.h5")
            rnn_model.save("tf_rnn_model.h5")
            cnn_model.save("tf_cnn_model.h5")
            print("-----------------------------------------")
        os.remove("taset_final.csv")
        break