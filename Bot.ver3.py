import os
from iqoptionapi.stable_api import IQ_Option
from configparser import ConfigParser
import pandas as pd
import datetime
import talib as tb
from colorama import init, Fore, Back
import time
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
import torch

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
markets_dayly_path = 'data/0000.markets.dayly.csv'
if os.path.exists(taset_finalD_path):
    print("Delete Old Data")
    os.remove('data/0.Train_data.csv')
    os.remove('data/0000.FulldataAmout.csv')
    if os.path.exists(markets_dayly_path):
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
    # --------------------------------------
    while True:
        print("------------------Date-------------------")
        ddtt = datetime.datetime.now()
        Timelistrun = ["00", "04", "07"]
        Truedaynow = ddtt.strftime("%A")
        Timeindaynow = ddtt.strftime("%H")
        print(Truedaynow, Timeindaynow, ddtt)
        if Timeindaynow in Timelistrun:
            Min_sec = 30 * 60
            time.sleep(Min_sec)
        else:
            break
        break
    # --------------------------------------
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
    Loop_get_candles = 60
    can_get = 916
    for i in range(Loop_get_candles):
        data = Iq.get_candles(Actives, 60, can_get, end_from_time)
        ANS = data + ANS
        end_from_time = int(data[0]["from"]) - 1
    df = pd.DataFrame(ANS)
    taset_df = df[['id', 'from', 'at', 'to', 'open', 'close', 'max', 'min', 'volume']].copy()
    taset_df.columns = ['Id', 'From', 'At', 'To', 'Open', 'Close', 'High', 'Low', 'Volume']

    print('LODDING...TECHNICHEN...')
    o = taset_df['Open'].values
    c = taset_df['Close'].values
    h = taset_df['High'].values
    l = taset_df['Low'].values
    v = taset_df['Volume'].astype(float).values
    ta = pd.DataFrame()
    ta["AVGPRICE"] = tb.AVGPRICE(o, h, l, c)
    ta["MEDPRICE"] = tb.MEDPRICE(h, l)
    ta["TYPPRICE"] = tb.TYPPRICE(h, l, c)
    ta["WCLPRICE"] = tb.WCLPRICE(h, l, c)
    # RSI
    ta['RSI14'] = tb.RSI(c, timeperiod=14)
    # Stochastic
    ta['STOCH14L'], ta['STOCH14R'] = tb.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3,slowd_matype=0)
    ta['STOCHF14L'], ta['STOCHF14R'] = tb.STOCHF(h, l, c, fastk_period=14, fastd_period=3, fastd_matype=0)
    ta['STOCHRSI14L'], ta['STOCHRSI14R'] = tb.STOCHRSI(c, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    ta['STOCHRSI3L'], ta['STOCHRSI3R'] = tb.STOCHRSI(c, timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0)
    # CCI - CommodityChannelIndex
    ta['CCI20'] = tb.CCI(h, l, c, timeperiod=20)
    # ADX - AverageDirectionalMovementIndex
    ta['ADX14'] = tb.ADX(h, l, c, timeperiod=14)
    # ADXR - Average Directional MovementIndex Rating
    ta['ADXR14'] = tb.ADXR(h, l, c, timeperiod=14)
    # MOM - Momentum
    ta['MOM'] = tb.MOM(c, timeperiod=10)
    # MACD - Moving Average Convergence / Divergence
    ta['MACD'], ta['MACDsignal'], ta['MACDhist'] = tb.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    ta['MACDFIX'], ta['MACDsignalFIX'], ta['MACDhistFIX'] = tb.MACDFIX(c, signalperiod=9)
    # WILLR - Williams' %R
    ta['WILLR'] = tb.WILLR(h, l, c, timeperiod=14)
    # ULTOSC - UltimateOscillator
    ta['ULTOSC'] = tb.ULTOSC(h, l, c, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    #  Simple Moving Average
    ta['SMA5'] = tb.SMA(c, timeperiod=5)
    ta['SMA10'] = tb.SMA(c, timeperiod=10)
    ta['SMA20'] = tb.SMA(c, timeperiod=20)
    ta['SMA30'] = tb.SMA(c, timeperiod=30)
    ta['SMA50'] = tb.SMA(c, timeperiod=50)
    ta['SMA100'] = tb.SMA(c, timeperiod=100)
    ta['SMA200'] = tb.SMA(c, timeperiod=200)
    # EMA - Exponential Moving Average
    ta['EMA5'] = tb.SMA(c, timeperiod=5)
    ta['EMA10'] = tb.SMA(c, timeperiod=10)
    ta['EMA20'] = tb.SMA(c, timeperiod=20)
    ta['EMA30'] = tb.SMA(c, timeperiod=30)
    ta['EMA50'] = tb.SMA(c, timeperiod=50)
    ta['EMA100'] = tb.SMA(c, timeperiod=100)
    ta['EMA200'] = tb.SMA(c, timeperiod=200)
    # WMA - WeightedMovingAverage
    ta['WMA20'] = tb.WMA(c, timeperiod=20)
    # MA - Moving average
    ta['MA9'] = tb.MA(c, timeperiod=9)

    selected_columns = ta.copy()
    taset_df = pd.concat([taset_df[['Id', 'From', 'At', 'To', 'Open', 'Close', 'High', 'Low', 'Volume']], selected_columns], axis=1)
    taset_final = taset_df.fillna(0)
    new_data = pd.DataFrame(taset_final.copy())
    taset_finalD_path = 'data/0.Train_data.csv'
    if os.path.exists(taset_finalD_path):
        taset_finalD = pd.read_csv(taset_finalD_path)
    else:
        taset_finalD = pd.DataFrame()
    taset_finalD = pd.concat([taset_finalD, new_data], ignore_index=True)
    taset_finalD.to_csv(taset_finalD_path, index=False)
    taset_finalD.to_csv('data/0000.FulldataAmout.csv', index=False)
    taset_row = taset_finalD.shape[0]
    taset_calum = taset_finalD.shape[1]
    taset_values = taset_calum * taset_row

    while True:
        rael_money = Money
        print("--------------Loadding Brain----------------")
        # โหลดข้อมูลจาก CSV
        data = pd.read_csv('data/0.Train_data.csv')

        features_data = data.drop(['Id', 'From', 'At', 'To'], axis=1)
        features = features_data.values

        dates = data['Id'].values

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        input_ids = []
        attention_masks = []

        for feature in features:
            encoded_dict = tokenizer.encode_plus(
                str(feature),
                add_special_tokens=True,
                max_length=64,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        dataset = TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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
        # ta["AVGPRICE"] = tb.AVGPRICE(o, h, l, c)
        # ta["MEDPRICE"] = tb.MEDPRICE(h, l)
        # ta["TYPPRICE"] = tb.TYPPRICE(h, l, c)
        # ta["WCLPRICE"] = tb.WCLPRICE(h, l, c)
        # # RSI
        # ta['RSI14'] = tb.RSI(c, timeperiod=14)
        # # Stochastic
        # ta['STOCH14L'], ta['STOCH14R'] = tb.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowk_matype=0,
        #                                           slowd_period=3, slowd_matype=0)
        # ta['STOCHF14L'], ta['STOCHF14R'] = tb.STOCHF(h, l, c, fastk_period=14, fastd_period=3, fastd_matype=0)
        # ta['STOCHRSI14L'], ta['STOCHRSI14R'] = tb.STOCHRSI(c, timeperiod=14, fastk_period=5, fastd_period=3,
        #                                                    fastd_matype=0)
        # ta['STOCHRSI3L'], ta['STOCHRSI3R'] = tb.STOCHRSI(c, timeperiod=3, fastk_period=3, fastd_period=14,
        #                                                  fastd_matype=14)
        # # CCI - CommodityChannelIndex
        # ta['CCI20'] = tb.CCI(h, l, c, timeperiod=20)
        # # ADX - AverageDirectionalMovementIndex
        # ta['ADX14'] = tb.ADX(h, l, c, timeperiod=14)
        # # ADXR - Average Directional MovementIndex Rating
        # ta['ADXR14'] = tb.ADXR(h, l, c, timeperiod=14)
        # # MOM - Momentum
        # ta['MOM'] = tb.MOM(c, timeperiod=10)
        # # MACD - Moving Average Convergence / Divergence
        # ta['MACD'], ta['MACDsignal'], ta['MACDhist'] = tb.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
        # ta['MACDFIX'], ta['MACDsignalFIX'], ta['MACDhistFIX'] = tb.MACDFIX(c, signalperiod=9)
        # # WILLR - Williams' %R
        # ta['WILLR'] = tb.WILLR(h, l, c, timeperiod=14)
        # # ULTOSC - UltimateOscillator
        # ta['ULTOSC'] = tb.ULTOSC(h, l, c, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        # #  Simple Moving Average
        # ta['SMA5'] = tb.SMA(c, timeperiod=5)
        # ta['SMA10'] = tb.SMA(c, timeperiod=10)
        # ta['SMA20'] = tb.SMA(c, timeperiod=20)
        # ta['SMA30'] = tb.SMA(c, timeperiod=30)
        # ta['SMA50'] = tb.SMA(c, timeperiod=50)
        # ta['SMA100'] = tb.SMA(c, timeperiod=100)
        # ta['SMA200'] = tb.SMA(c, timeperiod=200)
        # # EMA - Exponential Moving Average
        # ta['EMA5'] = tb.SMA(c, timeperiod=5)
        # ta['EMA10'] = tb.SMA(c, timeperiod=10)
        # ta['EMA20'] = tb.SMA(c, timeperiod=20)
        # ta['EMA30'] = tb.SMA(c, timeperiod=30)
        # ta['EMA50'] = tb.SMA(c, timeperiod=50)
        # ta['EMA100'] = tb.SMA(c, timeperiod=100)
        # ta['EMA200'] = tb.SMA(c, timeperiod=200)
        # # WMA - WeightedMovingAverage
        # ta['WMA20'] = tb.WMA(c, timeperiod=20)
        # # MA - Moving average
        # ta['MA9'] = tb.MA(c, timeperiod=9)
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
        # predictions = make_predictions(best_model, X_test)
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
        # os.remove('data/0.Train_data.csv')
        # os.remove('data/0.Decision_data.csv')
        # os.remove('data/0.PATTERN_data.csv')
        break