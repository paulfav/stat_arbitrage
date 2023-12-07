#Importing libraries

import pandas as pd
import requests
import config as cfg
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yfinance as yf
import os

from statsmodels.tsa.stattools import adfuller
from itertools import combinations
from joblib import Parallel, delayed
import multiprocessing
from collections import Counter

#############################################################################################################
#Getting the data


class Data:

    def __init__(self):
        self.sp500_tickers = self.get_sp500_tickers()
        self.all_stocks = self.get_stocks_data()
        

    def get_sp500_tickers(self):
        #getting SP500 tickers from wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url)
        df = pd.read_html(response.text, header=0)[0]
        tickers = df['Symbol'].tolist()
        pd.DataFrame(tickers).to_csv('sp500_tickers.csv')
        return tickers



    def get_stocks_data(self):
        try  : sp500_tickers = pd.read_csv('sp500_tickers.csv')
        except: sp500_tickers = self.get_sp500_tickers()

        try : all_stocks = pd.read_csv('sp500_close_price.csv')

        except:
            # we have to download the data
            all_stocks =pd.DataFrame()
            for ticker in sp500_tickers:
                #keeping just the close price
                stock = yf.download(ticker, start=cfg.start, end=cfg.end, progress=False)
                stock = stock['Close']
                stock = pd.DataFrame(stock)
                stock.columns = [ticker]
                #concatenating all the stocks
                all_stocks = pd.concat([all_stocks, stock], axis=1)
            all_stocks.to_csv('sp500_close_price.csv')
            all_stocks.replace([np.inf, -np.inf], np.nan, inplace=True)
            all_stocks.fillna(method='ffill', inplace=True)
            all_stocks.fillna(method='bfill', inplace=True)

        return all_stocks


#############################################################################################################

def test_cointegration(pair, black_list, data_window):
    col1, col2 = pair
    if black_list[col1] > 100 or black_list[col2] > 100:
        return None
    try : 
        residuals = data_window[col1] - data_window[col2]

        adf_result = adfuller(residuals)
        if adf_result[1] < cfg.p_val_threshold:
            mean = residuals.mean()
            std_dev = residuals.std()
            #print('found pair', col1, col2)
            return (col1, col2, {'p_val': adf_result[1], 'mean': mean, 'std_dev': std_dev})
    except:
        #print(f'error for pair {col1} - {col2}')
        black_list[col1] += 1
        black_list[col2] += 1
    return None


#############################################################################################################
#TRAINING THE MODEL

class Pair:
    def __init__(self, pair, black_list, data_window):
        self.pair = pair
        self.black_list = black_list
        self.data_window = data_window
        self.stock1, self.stock2 = pair
        self.mean = None
        self.std_dev = None
        pass


    def test_cointegration(self):
    
        if self.black_list[self.stock1] > 100 or self.black_list[self.stock2] > 100:
            return None
        try : 
            residuals = self.data_window[self.stock1] - self.data_window[self.stock2]

            adf_result = adfuller(residuals)
            if adf_result[1] < cfg.p_val_threshold:
                self.mean = residuals.mean()
                self.std_dev = residuals.std()
                #print('found pair', col1, col2)
                return (self.stock1, self.stock2, {'p_val': adf_result[1], 'mean': self.mean, 'std_dev': self.std_dev})
        except:
            #print(f'error for pair {col1} - {col2}')
            self.black_list[self.stock1] += 1
            self.black_list[self.stock2] += 1
        return None
    





class Trades(Data):
    def __init__(self, capital_ini=10_000, ):
        self.trades = []
        self.capital = capital_ini
        self.cash = 100_000
        self.capital_per_trade = 0
        self.capital_per_pair = 0
        self.best_pairs = []
        self.data_window = None

        pass




    def enter_position(self, pair, direction, entry_prices, size): 
        propor_on_1 = entry_prices[1] / (entry_prices[0] + entry_prices[1])
        propor_on_2 = 1 - propor_on_1
        self.cash -= size









    
                             
class Trade_today(Trades):
    def __init__(self):
        self.data_today = None
        self.z_today = None
        self.position = None
        self.pair = None

    def get_data_today(self, data_window, i):
        self.data_today = data_window.iloc[i]
        self.z_today = self.get_z_score_today()
        return self.data_today, self.z_today

    








def enter_position(trade_data_, z_score, stock1, stock2, i, ):

    entry_prices = (trade_data_[stock1][i], trade_data_[stock2][i])
    p1_0, p2_0 = entry_prices
    propor_on_1 = p2_0 / (p1_0 + p2_0)
    propor_on_2 = p1_0 / (p1_0 + p2_0)
    if  z_score < -cfg.enter_z: # long position
        direction = 'short'
        return direction, entry_prices, propor_on_1, propor_on_2


    elif  z_score > cfg.enter_z: # short position
        direction = 'long'
        entry_prices = (trade_data_[stock1][i], trade_data_[stock1][i])
        return direction, entry_prices, propor_on_1, propor_on_2

    return None



def exit_position(trade_data_, z_score, stock1, stock2, i, position, capital_per_trade, trades, forced_exit=False):
    direction, entry_prices = position
    entry_z_score = trades[-1]['z_score_entry']
    p1_0, p2_0 = entry_prices
    propor_on_1 = p2_0 / (p1_0 + p2_0)
    propor_on_2 = p1_0 / (p1_0 + p2_0)
    p1_final, p2_final = trade_data_[stock1][i], trade_data_[stock2][i]
    if direction == 'long':
        if z_score > -cfg.exit_z or forced_exit or z_score-entry_z_score < -cfg.stop_loss_z:
                profit = ((p1_final - p1_0)  * propor_on_1 - (p2_final - p2_0) * propor_on_2) * capital_per_trade
                return profit

    elif direction== 'short':
        if z_score  < cfg.exit_z or forced_exit or z_score-entry_z_score > cfg.stop_loss_z:
                profit = (-(p1_final - p1_0)  * propor_on_1 + (p2_final - p2_0) * propor_on_2) * capital_per_trade
                return profit
        

def update_trades(trade_data, z_score, stock1, stock2, i, position, event, trades, profit = None) :
    #print('event')
    if event == 'enter': 
        trades.append({'enter': trade_data.iloc[i]['Date'], 
                           'stock1': stock1, 
                           'stock2': stock2, 
                           'direction': position[0],
                           'z_score_entry': z_score, 
                           'forced_exit': False})
    else :
            trades[-1]['exit'] = trade_data.iloc[i]['Date']
            trades[-1]['P&L'] = profit
            trades[-1]['winning'] = profit > 0
            trades[-1]['z_score_exit'] = z_score
          

    if event == 'forced exit':
        trades[-1]['forced_exit'] = True


    return trades


pair_to_position ={}


def trade_at_date(data_today, z_today, i, cash, pair_to_position, capital_per_trade, best_pairs):
    for pair in best_pairs:
        stock1, stock2 = pair
        data_1, data_2 = data_today[stock1], data_today[stock2]
        residuals = data_1 - data_2
        if pair in pair_to_position:
            position = pair_to_position[pair]
            direction, entry_prices, propor_on_1, propor_on_2 = position
            if pair_pnl not in position:
                if direction == 'long': 
                    pair_pnl = [{'date': data_today['Date'], 'pnl':  }]
     
    #tradable_volume_today : risk management





    


def trade(trades, pair, cap_per_pair, trade_data, dict_of_pairs):
    stock1, stock2 = pair
    residuals = trade_data[stock1] - trade_data[stock2]
    #print(residuals)
    mean = dict_of_pairs[(stock1, stock2)]['mean']
    std_dev = dict_of_pairs[(stock1, stock2)]['std_dev']
    z_score_series = ((residuals - mean) / std_dev).fillna(0)
    position = None

    cap_per_trade = cap_per_pair * 0.02

    for i in range(len(z_score_series)):
       # print(z_score_series)
        z_score = z_score_series[i]
        if position is None and i <= len(z_score_series)* 0.9 :
            position = enter_position(trade_data, z_score, stock1, stock2, i)
            if position is not None:
                trades = update_trades(trade_data, z_score, stock1, stock2, i, position, 'enter', trades, None)

        elif position is not None  : 
            profit = None
            profit = exit_position(trade_data, z_score, stock1, stock2, i, position, cap_per_trade,  trades, forced_exit=False)
            if profit is not None:
                trades = update_trades(trade_data, z_score, stock1, stock2, i, position, 'exit', trades, profit)
                position = None
    
    #close remaining positions
    if position is not None:
        profit = exit_position(trade_data, z_score, stock1, stock2, i, position, cap_per_trade,  trades, forced_exit=True)
        trades = update_trades(trade_data, z_score, stock1, stock2, i, position, 'forced_exit', trades, profit)
            
    return trades





def main(): 

    window_calib = 60
    window_trade = 30
    capital = 100_000

    stock_data = get_stocks_data()

    line = window_calib
    while line < len(stock_data) - window_trade : 
        #we iterate over the trading periods (one month at a time)

        black_list = Counter()
        data_window = stock_data.iloc[line - window_calib : line]
        pairs = list(combinations(data_window.columns, 2))#[:1000]
        results = Parallel(n_jobs=cfg.num_cores)(delayed(test_cointegration)(pair, black_list, data_window) for pair in pairs)
        dict_of_pairs = {}
        for result in results:
            if result is not None:
                stock1, stock2, data = result
                pair = (stock1, stock2)
                dict_of_pairs[pair] = data

        if not dict_of_pairs:
            line += window_trade
            print("no pairs found for month")
            continue

        best_pairs = sorted(dict_of_pairs.keys(), key=lambda x: dict_of_pairs[x]['p_val'])[:min(len(dict_of_pairs), 100)]
        trade_data = stock_data.iloc[line: line + window_trade].copy().reset_index(drop=True)
        #print(trade_data)
        cap_per_pair = min((capital / len(best_pairs))*cfg.risk_per_month, capital*cfg.risk_per_pair)
        trades = []
        if capital <= 0:
            print('no capital left')
            break

        for pair in best_pairs:
            trades = trade(trades, pair, cap_per_pair, trade_data, dict_of_pairs)
        #sort trades by date of enter
        trades = sorted(trades, key=lambda x: x['enter'])
        trades_df = pd.DataFrame(trades)
        try : 
            PNL = trades_df['P&L'].sum()
        except:
            line += window_trade
            print("no trades found for month")
            continue
        #save with the month in the name
        period = pd.to_datetime(stock_data.iloc[line]['Date']).strftime("%Y-%m")
        os.makedirs(f'./trade_data/', exist_ok=True)
        os.makedirs(f'./trade_data/trades_{period}', exist_ok=True)
        trades_df.to_csv(f'./trade_data/trades_{period}/trades.csv')
      
        capital += PNL
        key_rationals = {'winning_rate': trades_df['winning'].mean(),
                        'profit_per_trade': trades_df['P&L'].mean(),
                        'max_drawdown': trades_df['P&L'].cumsum().min(),
                        'sharpe_ratio': trades_df['P&L'].mean() / trades_df['P&L'].std(), 
                        'PNL': PNL, 
                        'end_of_month_capital': capital, 
                        'capital_per_pair': cap_per_pair}
        key_rationals = pd.DataFrame(key_rationals, index=[0])
        key_rationals.to_csv(f'./trade_data/trades_{period}/key_rationals.csv')
        line += window_trade
        if capital <= 0:
            
            print('no capital left')
            break


## TODO:
# - add a stop loss
# - add a trailing stop
# - add a trailing take profit
# change trade stucture : iterate through time and at each time step, iterate through each par, instead of the opposite
# - add a risk management system

        



if __name__ == '__main__':

    main()







