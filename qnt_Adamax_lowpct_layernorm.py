from pandas import options, Series, set_option, read_csv, errors, DataFrame, to_datetime, Grouper, to_numeric, Timestamp
from warnings import simplefilter, filterwarnings, catch_warnings
simplefilter(action = "ignore", category = errors.PerformanceWarning)
from numpy import nanmedian, delete, maximum, isin, concatenate, cumprod, count_nonzero, unique, dot, corrcoef, log, arange, isnan, nanmax, nanmin, NaN, zeros, clip, where, float32, isfinite, array, append, NINF, intersect1d, setdiff1d, union1d
from numpy import nonzero as npnz
from math import ceil
from time import time
from sys import path, exit
from os import environ, mkdir
environ['API_KEY'] = "928a14b0-3442-4fc4-b082-945bfc7df464"
environ['DATA_BASE_URL'] = 'https://data-api.quantiacs.io/'
environ['CACHE_RETENTION'] = '7'
environ['CACHE_DIR'] = 'data-cache'
environ['ENGINE_CORRELATION_URL'] = 'https://quantiacs.io/referee/submission/forCorrelation'
environ['STATAN_CORRELATION_URL'] = 'https://quantiacs.io/statan/correlation'
environ['PARTICIPANT_ID'] = '0'
environ['OUTPUT_PATH'] = '/root/fractions.nc.gz'
environ['NONINTERACT'] = 'True'
if 'CLOUDSDK_CONFIG' in environ:
    env = 'colab'
    adds = ['/content/drive/MyDrive/stocks/',
            '/content/drive/MyDrive/2016_2019/']
    for item in adds:
        path.insert(1, item)
elif 'KAGGLE_URL_BASE' in environ:
    env = 'kaggle'
else:
    env = 'nb'
from os.path import isdir
from datetime import datetime, timedelta
from joblib import dump, load, Parallel, delayed
from json import load as jload
from json import loads as jloads
from copy import deepcopy
from decimal import Decimal
import xarray as xr
import qnt.data as qndata
#import qnt.output as qnout
import qnt.stats as qnstats
import qnt.xr_talib as xrta
import qnt.ta as qnta
from scipy.stats import yeojohnson, shapiro
#from gc import collect
#import torch as T
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import log_softmax
from torch import Tensor, stack, pow, exp, from_numpy, topk, no_grad, save, equal
from torch import arange as tarange
#from torch import float32 as tfloat32
from torch import cumsum as tcumsum
from torch import div as tdiv
from torch import where as twhere
#from torch import load as tload
from torch import cat as tcat
from torch import set_num_threads, set_num_interop_threads
from torch.optim import Adamax
import torch.multiprocessing as mp
from torch import isfinite as tisfinite
import matplotlib.pyplot as plt
from matplotlib import ticker
from optuna import create_study
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history
from optuna.trial import TrialState
from optuna import logging
from dropbox import Dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError
#from talib import CCI
import requests
from time import sleep
#import pytorch_optimizer as popt
#_dynamo.config.suppress_errors = True
#warnings.filterwarnings("ignore")
options.mode.chained_assignment = None
set_option("display.max_columns", None)
set_option("display.max_rows", None)
logging.set_verbosity(logging.ERROR)
def compare_return(data: DataFrame, close_name1: str, close_name2: str, interval: str | int, view=True):
    time_lt = []
    data.reset_index(inplace = True, drop = True)
    if type(data.loc[0, 'date']) == str:
        for idx in range(len(data)):
            time_lt.append(data.loc[idx, 'date'])
    else:
        for idx in range(len(data)):
            time_lt.append(data.loc[idx, 'date'].strftime('%Y%m%d'))
    def format_date(x, pos=None):
        input = clip(int(x + 0.5), 0, len(time_lt) - 1)
        real_val = time_lt[input]
        return real_val
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(24, 15))
    k_fraction, bottom_space, vol_fraction, blank, left_space, right_space, top_space = 0.5, 0.05, 0.1, 0.01, 0.07, 0.05, 0.05
    sub_fraction = (1 - bottom_space - k_fraction -
                    vol_fraction - blank * 3) / 2
    ax_ret = fig.add_axes([left_space, 1 - k_fraction, 1 -
                        left_space - right_space, k_fraction - top_space])
    ax_dd = fig.add_axes([left_space, bottom_space + vol_fraction + sub_fraction +
                        2 * blank, 1 - left_space - right_space, sub_fraction], sharex=ax_ret)
    ax_ind = fig.add_axes([left_space, bottom_space + vol_fraction +
                        blank, 1 - left_space - right_space, sub_fraction], sharex=ax_ret)
    ax_pos = fig.add_axes([left_space, bottom_space, 1 -
                        left_space - right_space, vol_fraction], sharex=ax_ret)
    width = 0.7 if interval  == 'All' else 1.8
    ax_ret.plot(time_lt, data[close_name1], linewidth = width, label = close_name1, color = '#f75000')
    ax_ret.plot(time_lt, data[close_name2], linewidth = width, label = close_name2, color = '#149db8')
    ax_dd.plot(time_lt, data[f'{close_name1} DD'], linewidth = width, label=f'{close_name1} Drawdown', color = '#f75000')
    ax_dd.plot(time_lt, data[f'{close_name2} DD'], linewidth = width, label=f'{close_name2} Drawdown', color = '#149db8')
    red_bar = where(data['pos'] > 0, data['pos'], 0)
    green_bar = where(data['pos'] < 0, data['pos'], 0)
    ax_pos.bar(time_lt, red_bar, linewidth=0, width = 1, label = 'long', color = '#d9267a')
    ax_pos.bar(time_lt, green_bar, linewidth=0, width = 1, label = 'short', color = '#017a6c')
    ind_vals = data['indicator']
    ax_ind.plot(time_lt, ind_vals, linewidth = 0, label = 'indicator', color = '#000000')
    ax_ind.fill_between(time_lt, ind_vals, 0, where = (ind_vals > 0), color = '#d9267a')
    ax_ind.fill_between(time_lt, ind_vals, 0, where = (ind_vals < 0), color = '#017a6c')
    mdd_x, mdd_y = time_lt[to_numeric(data[f'{close_name2} DD']).argmin()], data[f'{close_name2} DD'].min()
    ax_dd.plot(mdd_x, mdd_y, marker = '.', color = '#000000')
    ax_dd.annotate(text = f'{round(mdd_y, 2)}%', fontsize = 15, xycoords = 'data', xy = (mdd_x, mdd_y), xytext = (mdd_x, Decimal(mdd_y) * Decimal('0.85')))
    max_ind_x, max_ind_y = time_lt[to_numeric(data['indicator']).argmax()], data['indicator'].max()
    ax_ind.plot(max_ind_x, max_ind_y, marker = '.', color = '#000000')
    ax_ind.annotate(text = f'{round(max_ind_y, 2)}', fontsize = 15, xycoords = 'data', xy = (max_ind_x, max_ind_y), xytext = (max_ind_x, Decimal(max_ind_y) * Decimal('1.01')))
    plt.setp(ax_ret.get_xticklabels(), visible = False)
    plt.setp(ax_dd.get_xticklabels(), visible = False)
    plt.setp(ax_ind.get_xticklabels(), visible = False)
    plt.setp(ax_pos.get_xticklabels(), visible = True)
    for (ax, name) in ((ax_ret, 'Return(%)'), (ax_dd, 'Drawdown(%)'), (ax_ind, 'Indicator'), (ax_pos, 'Position')):
        ax.set_ylabel(ylabel = name, fontsize = 16)
        ax.tick_params(axis = 'y', labelsize = 14)
    ax_pos.tick_params(axis = 'x', labelsize = 14)
    ax_ret.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    ax_dd.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    ax_ind.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    ax_pos.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    ax_pos.xaxis.set_major_locator(locator = ticker.MultipleLocator(len(time_lt) // 9))
    #axv.set_xlabel(fontdict={'fontsize': 'x-small'})
    ax_ret.legend(framealpha = 0, markerscale = 0.8, fontsize = 15)
    ax_dd.legend(framealpha = 0, markerscale = 0.8, fontsize = 15)
    ax_ind.legend(framealpha = 0, markerscale = 0.8, fontsize = 12)
    ax_pos.legend(framealpha = 0, markerscale = 0.8, fontsize = 12)
    ax_ret.set_title(label = f'{close_name1} vs {close_name2.title()}({interval})', fontsize = 22)
    #plt.savefig(filename, dpi = 280)
    # plt.get_current_fig_manager().window.state('zoomed')
    if view:
        plt.show()
    plt.close('all')
def upload_data(dbx, local_file_path, dropbox_path):
    with open(local_file_path, 'rb') as f:
        try:
            dbx.files_upload(f.read(), dropbox_path,
                             mode=WriteMode('overwrite'))
        except ApiError as err:
            # This checks for the specific error where a user doesn't have enough Dropbox space quota to upload this file
            if (err.error.is_path() and
                    err.error.get_path().error.is_insufficient_space()):
                exit("ERROR: Cannot back up; insufficient space.")
            elif err.user_message_text:
                print(err.user_message_text)
                exit()
            else:
                print(err)
                exit()
def download_data(dbx, local_file_path, dropbox_file_path):
    try:
        with open(local_file_path, 'wb') as f:
            metadata, result = dbx.files_download(path=dropbox_file_path)
            f.write(result.content)
        return local_file_path
    except Exception as err:
        return 1
def process_column_shapiro(col_data, valid_mask_col, non_zero_range_col):
    results_col = []
    final_valid_mask = valid_mask_col & non_zero_range_col
    for window, is_valid in zip(col_data, final_valid_mask):
        if not is_valid:
            results_col.append(NaN)
            continue
        try:
            stat = shapiro(window[~isnan(window)])[0]
            results_col.append(stat)
        except ValueError as ve:
            # Shapiro might raise ValueError if sample size is too small after NaN removal
            # print(f"ValueError calculating Shapiro for a window: {ve}") # Optional: log error
            results_col.append(NaN)
        except Exception as e:
            # Catch other potential errors
            # print(f"Error calculating Shapiro for a window: {e}") # Optional: log error
            results_col.append(NaN)
    return results_col
def roll_Shapiro(max_na_cnt, cal_array, window_indices):
    windows = cal_array[window_indices]
    nan_counts = isnan(windows).sum(axis = 1)
    with catch_warnings():
        filterwarnings("ignore", message = ".*All-NaN slice encountered.*", category = RuntimeWarning)
        ranges = nanmax(windows, axis = 1) - nanmin(windows, axis = 1)
        valid_mask_nan = (nan_counts <= max_na_cnt)
        non_zero_range = (ranges > 0)
    results_list = Parallel(n_jobs = -1, backend = "loky")(
        delayed(process_column_shapiro)(windows[:, :, i], valid_mask_nan[:, i], non_zero_range[:, i])
        for i in range(assets_cnt)
    )
    return DataFrame(array(results_list).T, index = stk_chg_inds, columns = assets) * valid_liq
def wma(df, n):
    weights = arange(1, n + 1)
    wmas = df[assets].rolling(n).apply(lambda x: dot(x, weights) / weights.sum(), raw=True)
    return wmas
def cal_corr(pr_data, window):
    pr_values = pr_data.values
    result_index = [i - 200 for i in raw_stk_chg_inds]
    def _process_index(idx):
        i = raw_stk_chg_inds[idx]
        window_data = pr_values[i - window + 1: i + 1, :]
        corr_matrix = corrcoef(window_data, rowvar = False)
        corr_sums = where(corr_matrix < 0, corr_matrix, 0).sum(axis = 0)
        return corr_sums
    counts_list = Parallel(n_jobs = -1)(
        delayed(_process_index)(i) for i in range(len(raw_stk_chg_inds))
    )
    return DataFrame(array(counts_list), index = result_index, columns = assets) * valid_liq
def get_features(returns_arr, vol_arr):
    if returns_arr.shape != vol_arr.shape:
        raise ValueError("Price changes and volumes must have the same shape")
    stat_dict = dict()
    periods_dict = {'std_lwma': [5, 90],
    'logret': [10, 180], 'dd': [55],
    'up': [20], 'ADOSC': [50], 'NATR': [15],
    'aroon': [20, 140], 'money_vol': [40],
    'BOP': [120]}
    roll_size = 60
    max_na_cnt = roll_size * 0.1
    window_indices = array([arange(i - (roll_size - 1), i + 1) for i in raw_stk_chg_inds])
    typ_price = xrta.TYPPRICE(data)
    day_logret = log(typ_price.to_pandas()).diff(1)
    day_logret.fillna(0, inplace=True)
    ret_mean = day_logret.rolling(100, min_periods = 100).mean()
    ret_std = day_logret.rolling(100, min_periods = 100).std()
    ret_zscore = (day_logret - ret_mean) / ret_std
    pos_1 = (1 <= ret_zscore) & (ret_zscore < 2)
    pos_2 = (2 <= ret_zscore) & (ret_zscore < 3)
    pos_3 = ret_zscore >= 3
    neg_1 = (-2 < ret_zscore) & (ret_zscore <= -1)
    neg_2 = (-3 < ret_zscore) & (ret_zscore <= -2)
    neg_3 = ret_zscore <= -3
    ret_class_df = DataFrame(index=day_logret.index, columns=assets)
    ret_class_df[assets] = where(pos_1, 1,
                               where(pos_2, 2,
                                    where(pos_3, 3,
                                         where(neg_1, -1,
                                              where(neg_2, -2,
                                                   where(neg_3, -3, 0))))))
    def calc_wmas():
        wma5 = wma(ret_class_df, 5)
        wma90 = wma(ret_class_df, 90)
        return wma5, wma90
    wma5_df, wma90_df = calc_wmas()
    stat_dict['std_lwma5'] = DataFrame(array(wma5_df.loc[stk_select_dates, :]),
                                      index=stk_chg_inds, columns=assets) * valid_liq
    stat_dict['std_lwma90'] = DataFrame(array(wma90_df.loc[stk_select_dates, :]),
                                       index=stk_chg_inds, columns=assets) * valid_liq
    for period in periods_dict['logret']:
        #roll_volat = array(day_logret.rolling(period, min_periods = period).sem())
        log_total_ret = array(day_logret.rolling(period, min_periods = period).sum())
        stat_dict[f'logret{period}'] = DataFrame(log_total_ret[raw_stk_chg_inds, :], index = stk_chg_inds, columns = assets)
        if period == 10:
            stat_dict[f'Shapiro_logret{period}'] = roll_Shapiro(max_na_cnt, log_total_ret, window_indices)
    for period in periods_dict['dd']:
        dd_df = pr_copy / pr_copy.rolling(period, min_periods = period).max() - 1
        cal_array = dd_df.to_numpy()
        stat_dict[f'Shapiro_dd{period}'] = roll_Shapiro(max_na_cnt, cal_array, window_indices)
    daily_up_df = day_logret.copy(deep = True)
    daily_up_df[assets] = where(day_logret > 0, 1, 0)
    for period in periods_dict['up']:
        up_sum = daily_up_df.rolling(period, min_periods = period).sum()
        stat_dict[f'Shapiro_up{period}'] = roll_Shapiro(max_na_cnt, up_sum.to_numpy(), window_indices)
    for period in periods_dict['ADOSC']:
        adosc_df = (xrta.ADOSC(data, 3, period) / qnta.wilder_ma(vol, period)).to_pandas()
        stat_dict[f'Shapiro_ADOSC3_{period}'] = roll_Shapiro(max_na_cnt, adosc_df.to_numpy(), window_indices)
    for period in periods_dict['NATR']:
        natr_df = xrta.NATR(data, period).to_pandas()
        stat_dict[f'NATR{period}'] = DataFrame(array(natr_df.loc[stk_select_dates, :]), index = stk_chg_inds, columns = assets) * valid_liq
    for period in periods_dict['aroon']:
        up, down = xrta.AROON(data, period)
        stat_dict[f'uparoon{period}'] = DataFrame(array(up.to_pandas().loc[stk_select_dates, :]), index = stk_chg_inds, columns = assets) * valid_liq
        if period == 20:
            stat_dict[f'downaroon{period}'] = DataFrame(array(down.to_pandas().loc[stk_select_dates, :]), index = stk_chg_inds, columns = assets) * valid_liq
    for period in periods_dict['money_vol']:
        money_vol = vol * prices
        total_money_vol = money_vol.sum(dim='asset')
        money_vol_share = money_vol / total_money_vol
        money_vol_df = qnta.lwma(money_vol_share, period).to_pandas()
        stat_dict[f'money_vol{period}'] = DataFrame(array(money_vol_df.loc[stk_select_dates, :]), index = stk_chg_inds, columns = assets) * valid_liq
    for period in periods_dict['BOP']:
        bop_trima = xrta.TRIMA(xrta.BOP(data), period).to_pandas()
        stat_dict[f'BOP_trima{period}'] = DataFrame(array(bop_trima.loc[stk_select_dates, :]), index = stk_chg_inds, columns = assets) * valid_liq
    stat_dict[f'corrsum_40'] = cal_corr(pr_copy, 40)
    return stat_dict
def transform_sector(new_data):
    mapping = {item: item if item in label_encoder.classes_ else 'Unknown' for item in new_data.unique()}
    mapped_series = new_data.map(mapping)
    encoded_values = label_encoder.transform(mapped_series)
    return Series(encoded_values, index=new_data.index)

def objective(trial):
    print('Trial:', trial.number)
    long_actor = trial.suggest_categorical("long_actor", func_lt)
    short_actor = trial.suggest_categorical("short_actor", func_lt)
    long_critic = trial.suggest_categorical("long_critic", func_lt)
    short_critic = trial.suggest_categorical("short_critic", func_lt)
    trader_wt_decay = 1 / 10 ** 3
    hidden_size = 2 ** 8
    total_epochs = 36
    stk_cnt_lt = [10, 16, 20, 25, 32, 40, 50]
    stk_cnt = trial.suggest_categorical("stk_cnt", stk_cnt_lt)
    stk_ratio = trial.suggest_int('stk_ratio', 80, 100, step = 5) / 100
    long_pct = 0.55
    long_cnt = ceil(stk_cnt * long_pct)
    short_cnt = stk_cnt - long_cnt
    stk_pct = stk_ratio / stk_cnt
    #batch_mom = trial.suggest_int("batch_mom", 10, 90, step = 40) / 100
    #batch_mom = 0.9
    long_trader_lr = 0.002
    short_trader_lr = 0.002
    #sector_dim = trial.suggest_int("sector_dim", 4, 8, step = 1)
    #sector_dim = 7
    #month_dim = trial.suggest_int("month_dim", 4, 8, step = 1)
    #month_dim = 7
    #weekday_dim = 3
    hold_period = trial.suggest_int('hold_period', 1, 4, step = 1)
    update_cnt = 6
    model_type = trial.suggest_categorical("model_type", ['gbdt', 'dart', 'rf', 'upper', 'lower'])
    atr_ma = trial.suggest_int("atr_ma", 10, 30, step = 5)
    longloss_N = -trial.suggest_int("long_loss_N", 15, 30, step = 5) / 10
    shortloss_N = -trial.suggest_int("short_loss_N", 15, 30, step = 5) / 10
    longgain_N = trial.suggest_int("long_gain_N", 20, 40, step = 5) / 10
    shortgain_N = trial.suggest_int("short_gain_N", 20, 40, step = 5) / 10
    strict_long = trial.suggest_categorical("strict_long", [True, False])
    strict_short = trial.suggest_categorical("strict_short", [True, False])
    long_min_days = trial.suggest_int("long_min_days", 0, 100, step = 5)
    short_min_days = trial.suggest_int("short_min_days", 0, 60, step = 5)
    long_lgbm = model_dict['long'][model_type]
    short_lgbm = model_dict['short'][model_type]
    long_upper = model_dict['long']['upper']
    long_lower = model_dict['long']['lower']
    short_upper = model_dict['short']['upper']
    short_lower = model_dict['short']['lower']
    long_eps = 1 / 10 ** 8
    short_eps = 1 / 10 ** 8
    #hidden_size: int, batch_mom: float, actor_act: nn, critic_act: nn, sector_dim: int, weekday_dim: int, month_dim: int
    long_model = Trader(hidden_size, actor_act = getattr(nn, long_actor), critic_act = getattr(nn, long_critic))
    short_model = Trader(hidden_size, actor_act = getattr(nn, short_actor), critic_act = getattr(nn, short_critic))
    long_opt = Adamax(long_model.parameters(),
    lr = long_trader_lr, eps = long_eps, weight_decay = trader_wt_decay, maximize = False, foreach = True)
    short_opt = Adamax(short_model.parameters(),
    lr = short_trader_lr, eps = short_eps, weight_decay = trader_wt_decay, maximize = False, foreach = True)
    params_dict = {'atr_ma': atr_ma,
                   'long':{'gain': longgain_N, 'loss': longloss_N,
                    'strict': strict_long, 'days': long_min_days,
                    'cnt': long_cnt, 'lgbm': long_lgbm,
                    'upper': long_upper, 'lower': long_lower,
                    'model': long_model, 'opt': long_opt},
                    'short': {'gain': shortgain_N, 'loss': shortloss_N,
                    'strict': strict_short, 'days': short_min_days,
                    'cnt': short_cnt, 'lgbm': short_lgbm,
                    'upper': short_upper, 'lower': short_lower,
                    'model': short_model, 'opt': short_opt}}

    return process(total_epochs, hold_period, update_cnt, stk_pct, params_dict)

def test_model(stk_pct, params_dict):
    long_model, short_model = params_dict['long']['model'], params_dict['short']['model']
    long_cnt, short_cnt = params_dict['long']['cnt'], params_dict['short']['cnt']
    long_lgbm, short_lgbm = params_dict['long']['lgbm'], params_dict['short']['lgbm']
    long_upper, long_lower = params_dict['long']['upper'], params_dict['long']['lower']
    short_upper, short_lower = params_dict['short']['upper'], params_dict['short']['lower']
    longgain_N, longloss_N = params_dict['long']['gain'], params_dict['long']['loss']
    shortgain_N, shortloss_N = params_dict['short']['gain'], params_dict['short']['loss']
    strict_long, strict_short = params_dict['long']['strict'], params_dict['short']['strict']
    long_min_days, short_min_days = params_dict['long']['days'], params_dict['short']['days']
    wts_df = prices_df.copy(deep = True)
    wts_df.loc[:, assets] = 0
    wts_np = zeros((row_cnt, assets_cnt))
    wts_np[:] = NaN
    atr_df = qnta.atr(high, low, prices, params_dict['atr_ma']).to_pandas()
    atr_df.reset_index(inplace = True)
    atr_df = atr_df[atr_df['time'] >= dt_start]
    atr_df.reset_index(inplace = True, drop = True)
    atr_np = atr_df[assets].to_numpy()
    #long_model.hold_indices = array([], dtype = int)
    #short_model.hold_indices = array([], dtype = int)
    select_params = {'long': {'logits': [], 'new_cnt': 0, 'lgbm': long_lgbm},
                     'short': {'logits': [], 'new_cnt': 0, 'lgbm': short_lgbm},
                     'df': []}
    long_model.eval()
    short_model.eval()
    with no_grad():
        for chg_date in all_tensors_dict.keys():
            nonliq_indices = nonliq_np_dict[chg_date]
            final_df = all_features_dict[chg_date]
            final_tensor = all_tensors_dict[chg_date]
            chg_date_idx = date_indices[chg_date]
            liq_tensor = liquid_tensor[chg_date_idx, :]
            if chg_date == fst_select_date:
                new_long_cnt, new_short_cnt = long_cnt, short_cnt
                long_policy = long_model(final_tensor, liq_tensor)[0]
                short_policy = short_model(final_tensor, liq_tensor)[0]
                if type(long_policy) != int and type(short_policy) != int:
                    long_logits = long_policy.view(-1)
                    short_logits = short_policy.view(-1)
                    select_params['long']['logits'] = long_logits
                    select_params['short']['logits'] = short_logits
                    select_params['long']['new_cnt'] = new_long_cnt
                    select_params['short']['new_cnt'] = new_short_cnt
                    select_params['df'] = final_df
                    new_long_indices, new_short_indices = selection(select_params)
                    LongSys = TradeSystem(new_long_indices, 'long', 'None', 0, stk_pct, longloss_N, longgain_N, 1, long_upper, long_lower, strict_long, long_min_days)
                    ShortSys = TradeSystem(new_short_indices, 'short', 'None', 0, -stk_pct, shortloss_N, shortgain_N, 1, short_upper, short_lower, strict_short, short_min_days)
                    #initial_stks, trade_type, add_type, max_add_cnt, stk_pct, loss_N, gain_N, add_N, rf_upper, rf_lower, strict, min_days
                    LongSys.hold_np[:, 0] = prices_np[chg_date_idx, new_long_indices]
                    ShortSys.hold_np[:, 0] = prices_np[chg_date_idx, new_short_indices]
                    LongSys.hold_np[:, 6] = LongSys.hold_np[:, 6].astype('int16')
                    ShortSys.hold_np[:, 6] = ShortSys.hold_np[:, 6].astype('int16')
                    wts_np[chg_date_idx, new_long_indices] = stk_pct
                    wts_np[chg_date_idx, new_short_indices] = -stk_pct
                else:
                    return 1, 1, 1, 1, 1, 1, 1
            else:
                LongSys.hold_np[:, 5] += 1
                ShortSys.hold_np[:, 5] += 1
                hold_long_indices, hold_short_indices = LongSys.hold_np[:, 6].astype('int16'), ShortSys.hold_np[:, 6].astype('int16')
                all_hold_indices = append(hold_long_indices, hold_short_indices)
                stop_long_cnt, stop_short_cnt = long_cnt - hold_long_indices.shape[0], short_cnt - hold_short_indices.shape[0]
                #加一個條件: 將那些在hold_long_stks中，而不在nonliq_stks中的股票傳入pred_quantile()中
                valid_long_indices = setdiff1d(hold_long_indices, nonliq_indices, assume_unique = True)
                valid_short_indices = setdiff1d(hold_short_indices, nonliq_indices, assume_unique = True)
                illiq_long_indices = intersect1d(hold_long_indices, nonliq_indices, assume_unique = True)
                illiq_short_indices = intersect1d(hold_short_indices, nonliq_indices, assume_unique = True)
                if valid_long_indices.shape[0] > 0:
                    dropped_long_names = LongSys.pred_quantile(final_df.loc[assets[valid_long_indices], :])
                    dropped_long_indices = [assets_map_dict[name] for name in dropped_long_names]
                    wts_np[chg_date_idx, dropped_long_indices] = 0
                    mask = isin(LongSys.hold_np[:, 6].astype('int16'), dropped_long_indices, invert = True)
                    LongSys.hold_np = LongSys.hold_np[mask]
                else:
                    dropped_long_names = []
                if valid_short_indices.shape[0] > 0:
                    dropped_short_names = ShortSys.pred_quantile(final_df.loc[assets[valid_short_indices], :])
                    dropped_short_indices = [assets_map_dict[name] for name in dropped_short_names]
                    wts_np[chg_date_idx, dropped_short_indices] = 0
                    mask = isin(ShortSys.hold_np[:, 6].astype('int16'), dropped_short_indices, invert = True)
                    ShortSys.hold_np = ShortSys.hold_np[mask]
                else:
                    dropped_short_names = []
                if illiq_long_indices.shape[0] > 0:
                    wts_np[chg_date_idx, illiq_long_indices] = 0
                    mask = isin(LongSys.hold_np[:, 6].astype('int16'), illiq_long_indices, invert = True)
                    LongSys.hold_np = LongSys.hold_np[mask]
                if illiq_short_indices.shape[0] > 0:
                    wts_np[chg_date_idx, illiq_short_indices] = 0
                    mask = isin(ShortSys.hold_np[:, 6].astype('int16'), illiq_short_indices, invert = True)
                    ShortSys.hold_np = ShortSys.hold_np[mask]
                new_long_cnt, new_short_cnt = len(dropped_long_names) + len(illiq_long_indices) + stop_long_cnt, len(dropped_short_names) + len(illiq_short_indices) + stop_short_cnt

                if new_long_cnt > long_cnt:
                    print('long_cnt:', long_cnt, 'new_long_cnt:', new_long_cnt)
                    print('Hold Long indices:', hold_long_indices, len(hold_long_indices))
                    print('Dropped long indices:', dropped_long_indices, len(dropped_long_indices))
                    print('Illiq long indices:', illiq_long_indices, len(illiq_long_indices))
                    raise ValueError('Stop long cnt:', stop_long_cnt)
                if new_short_cnt > short_cnt:
                    print('short_cnt:', short_cnt, 'new_short_cnt:', new_short_cnt)
                    print('Short indices:', hold_short_indices, len(hold_short_indices))
                    print('Dropped short indices:', dropped_short_indices, len(dropped_short_indices))
                    print('Illiq short indices:', illiq_short_indices, len(illiq_short_indices))
                    raise ValueError('Stop short cnt:', stop_short_cnt)

                long_policy = long_model(final_tensor, liq_tensor)[0]
                short_policy = short_model(final_tensor, liq_tensor)[0]
                if type(long_policy) != int and type(short_policy) != int:
                    long_logits = long_policy.view(-1)
                    short_logits = short_policy.view(-1)
                    long_logits[all_hold_indices] = -1
                    short_logits[all_hold_indices] = -1
                    select_params['long']['logits'] = long_logits
                    select_params['short']['logits'] = short_logits
                    select_params['long']['new_cnt'] = new_long_cnt
                    select_params['short']['new_cnt'] = new_short_cnt
                    select_params['df'] = final_df
                    new_long_indices, new_short_indices = selection(select_params)
                    if new_long_indices.shape[0] > 0:
                        new_np = zeros((new_long_indices.shape[0], 7), dtype = 'float64')
                        new_np[:, 0] = prices_np[chg_date_idx, new_long_indices]
                        new_np[:, 6] = new_long_indices
                        LongSys.hold_np = concatenate((LongSys.hold_np, new_np), axis = 0)
                        LongSys.hold_np[:, 6] = LongSys.hold_np[:, 6].astype('int16')

                        if len(LongSys.hold_np) > long_cnt:
                            print('New long indices:', new_long_indices, len(new_long_indices))
                            print('Initial hold long names:', hold_long_indices, len(hold_long_indices))
                            print('Dropped long indices:', dropped_long_indices, len(dropped_long_indices))
                            print('Illiq long indices:', illiq_long_indices, len(illiq_long_indices))
                            print('Stop long cnt:', stop_long_cnt)
                            print('Current hold indices:', LongSys.hold_np[:, 6])
                            raise ValueError('Current Long cnt:', len(LongSys.hold_np), 'Default Long cnt:', long_cnt)

                    if new_short_indices.shape[0] > 0:
                        new_np = zeros((new_short_indices.shape[0], 7), dtype = 'float64')
                        new_np[:, 0] = prices_np[chg_date_idx, new_short_indices]
                        new_np[:, 6] = new_short_indices
                        ShortSys.hold_np = concatenate((ShortSys.hold_np, new_np), axis = 0)
                        ShortSys.hold_np[:, 6] = ShortSys.hold_np[:, 6].astype('int16')

                        if len(ShortSys.hold_np) > short_cnt:
                            print('New short indices:', new_short_indices, len(new_short_indices))
                            print('Initial hold short names:', hold_short_indices, len(hold_short_indices))
                            print('Dropped short indices:', dropped_short_indices, len(dropped_short_indices))
                            print('Illiq short indices:', illiq_short_indices, len(illiq_short_indices))
                            print('Stop short cnt:', stop_short_cnt)
                            print('Current hold indices:', ShortSys.hold_np[:, 6])
                            raise ValueError('Current Short cnt:', len(ShortSys.hold_np), 'Default Short cnt:', short_cnt)

                    wts_np[chg_date_idx, LongSys.hold_np[:, 6].astype('int16')] = stk_pct
                    wts_np[chg_date_idx, ShortSys.hold_np[:, 6].astype('int16')] = -stk_pct
                else:
                    return 1, 1, 1, 1, 1, 1, 1
            trade_date_arr = monitor_dict.get(chg_date, [])
            if len(trade_date_arr) > 0:
                for trade_date in trade_date_arr:
                    LongSys.hold_np[:, 5] += 1
                    ShortSys.hold_np[:, 5] += 1
                    date_idx = date_indices[trade_date]
                    wts_np = LongSys.Monitor(date_idx, atr_np, prices_np, wts_np)
                    wts_np = ShortSys.Monitor(date_idx, atr_np, prices_np, wts_np)
        wts_df.loc[:, assets] = wts_np
        wts_df.loc[:, assets] = wts_df.loc[:, assets].ffill(axis = 0) * liquid_arr
        wts_df.fillna(0, inplace = True, axis = 0)
        wts_df.loc[:, assets] = wts_df.loc[:, assets].astype(float)
        wts_df.set_index('time', inplace = True)
        wts_df.columns.names = ['asset']
        wts_xr = wts_df.unstack().to_xarray()
        stats = qnstats.calc_stat(data, wts_xr)
        stats_df = stats.to_pandas()
        stats_df.reset_index(inplace = True)
        stats_df = stats_df[stats_df['time'] >= dt_start]
        stats_df.reset_index(inplace = True, drop = True)
        #print('stats_df:\n', stats_df.shape)
        sharpe, arr, mdd, volat = stats_df.loc[pr_last_idx, 'sharpe_ratio'], stats_df.loc[pr_last_idx, 'mean_return'], -stats_df.loc[pr_last_idx, 'max_drawdown'], stats_df.loc[pr_last_idx, 'volatility']
        #hold_time = stats_df.loc[pr_last_idx, 'avg_holding_time']
        calmar = arr / mdd
    return sharpe, arr, mdd, volat, calmar, stats_df, wts_df
def selection(select_params):
    long_logits, short_logits = select_params['long']['logits'], select_params['short']['logits']
    new_long_cnt, new_short_cnt = select_params['long']['new_cnt'], select_params['short']['new_cnt']
    long_lgbm, short_lgbm = select_params['long']['lgbm'], select_params['short']['lgbm']
    final_df = select_params['df']
    new_long_indices, new_short_indices = array([], dtype = int), array([], dtype = int)
    if new_long_cnt > 0:
        new_long_indices = topk(long_logits, k = new_long_cnt, largest = True, sorted = False)[1].numpy().astype(int)
    if new_short_cnt > 0:
        new_short_indices = topk(short_logits, k = new_short_cnt, largest = True, sorted = False)[1].numpy().astype(int)
    common_stks = intersect1d(new_long_indices, new_short_indices, assume_unique = True).astype(int)
    if common_stks.shape[0] > 0:
        common_names = assets[common_stks]
        long_rets = long_lgbm.predict(final_df.loc[common_names, :])
        short_rets = short_lgbm.predict(final_df.loc[common_names, :])
        mask = long_rets >= short_rets
        final_long_stks  = common_stks[mask]
        final_short_stks = common_stks[~mask]
        long_diff = setdiff1d(new_long_indices, common_stks, assume_unique = True)
        short_diff = setdiff1d(new_short_indices, common_stks, assume_unique = True)
        new_long_indices = append(long_diff, final_long_stks).astype(int)
        new_short_indices = append(short_diff, final_short_stks).astype(int)
        long_vacants, short_vacants = new_long_cnt - new_long_indices.shape[0], new_short_cnt - new_short_indices.shape[0]
        if short_vacants > 0:
            short_logits[new_long_indices] = -1
            short_logits[new_short_indices] = -1
            new_short = topk(short_logits, k = short_vacants, largest = True, sorted = False)[1].numpy()
            new_short_indices = append(new_short_indices, new_short).astype(int)
            long_logits[new_short] = -1
        if long_vacants > 0:
            long_logits[new_long_indices] = -1
            long_logits[new_short_indices] = -1
            new_long = topk(long_logits, k = long_vacants, largest = True, sorted = False)[1].numpy()
            new_long_indices = append(new_long_indices, new_long).astype(int)
    return new_long_indices, new_short_indices
class BestTrial():
    def __init__(self):
        self.score = -1e5
        self.sharpe = 0
        self.arr = 0
        self.mdd = -0.99
        self.vol = 0.1
        self.calmar = 0
        self.stats_df = []
        self.wts_df = []
        self.long_state_dict = dict()
        self.short_state_dict = dict()
        self.long_lrs = []
        self.short_lrs = []
        self.test_scores = []
        self.median_sharpe = 0
        self.stk_cnt = 10
        self.test_time_lt = []
        self.train_time_lt = []
        self.stat_time_lt = []
class TradeSystem():
    def __init__(self, initial_indices, trade_type, add_type, max_add_cnt, stk_pct, loss_N, gain_N, add_N, rf_upper, rf_lower, strict, min_days):
        self.max_add_cnt = max_add_cnt
        self.stk_pct = stk_pct
        self.loss_N = loss_N
        self.gain_N = gain_N
        self.add_N = add_N
        self.add_type = add_type
        if add_type == 'up':
            self.add_method = self.Up
        elif add_type == 'down':
            self.add_method = self.Down
        else:
            self.add_method = self.NoAdd
        self.cal_N = self.long_N if trade_type == 'long' else self.short_N
        self.rf_upper = rf_upper
        self.rf_lower = rf_lower
        self.min_days = min_days
        self.hold_df = DataFrame(0.00, columns = ['cost', 'last_pr', 'atr', 'N', 'add_cnt', 'days', 'index'], index = arange(len(initial_indices)), dtype = 'float64')
        #cost: 0, last_pr:1, atr: 2, N: 3, add_cnt: 4, days: 5, index: 6
        self.hold_df['index'] = initial_indices
        self.hold_np = self.hold_df.to_numpy()
        self.hold_np[:, 6] = self.hold_np[:, 6].astype('int16')
        if strict:
            self.pred_quantile = self.strict_same_direction_pred
        else:
            self.pred_quantile = self.same_direction_pred
    def Monitor(self, date_idx, atr_np, prices_np, wts_np):
        if self.hold_np.shape[0] > 0:
            stk_indices = self.hold_np[:, -1].astype('int16')
            self.hold_np[:, 1] = prices_np[date_idx, stk_indices]
            self.hold_np[:, 2] = atr_np[date_idx, stk_indices]
            self.hold_np[self.hold_np[:, 2] == 0, 2] = 1e-6
            self.hold_np[:, 3] = self.cal_N()
            loss_cond = self.hold_np[:, 3] <= self.loss_N
            gain_cond = self.hold_np[:, 3] >= self.gain_N
            min_cond = self.hold_np[:, 5] >= self.min_days
            matching_rows = (loss_cond | gain_cond) & min_cond
            stop_stks_row_indices = npnz(matching_rows)[0]
            if len(stop_stks_row_indices) > 0:
                stop_true_indices = self.hold_np[stop_stks_row_indices, 6].astype('int16')
                wts_np[date_idx, stk_indices] = self.stk_pct
                wts_np[date_idx, stop_true_indices] = 0
                self.hold_np = delete(self.hold_np, stop_stks_row_indices, axis = 0)
            else:
                wts_np[date_idx, stk_indices] = self.stk_pct
            return self.add_method(date_idx, wts_np)
        return wts_np
    def Up(self, date_idx, wts_np):
        add_cond = (self.hold_df['N'] >= self.add_N) & (self.hold_df['N'] < self.gain_N) & (self.hold_df['add_cnt'] < self.max_add_cnt)
        add_stks = self.hold_df[add_cond].index
        if len(add_stks) > 0:
            all_add_cnt = self.hold_df.loc[add_stks, 'add_cnt'] + 1
            self.hold_df.loc[add_stks, 'add_cnt'] = all_add_cnt
            self.hold_df.loc[add_stks, 'cost'] = all_add_cnt * self.hold_df.loc[add_stks, 'cost'] / (all_add_cnt + 1) + self.hold_df.loc[add_stks, 'last_pr'] / (all_add_cnt + 1)
            wts_np[date_idx, self.hold_df.loc[add_stks, 'index']] += self.stk_pct
        return wts_np
    def Down(self, date_idx, wts_np):
        add_cond = (self.hold_df['N'] <= -self.add_N) & (self.hold_df['N'] > self.loss_N) & (self.hold_df['add_cnt'] < self.max_add_cnt)
        add_stks = self.hold_df[add_cond].index
        if len(add_stks) > 0:
            all_add_cnt = self.hold_df.loc[add_stks, 'add_cnt'] + 1
            self.hold_df.loc[add_stks, 'add_cnt'] = all_add_cnt
            self.hold_df.loc[add_stks, 'cost'] = all_add_cnt * self.hold_df.loc[add_stks, 'cost'] / (all_add_cnt + 1) + self.hold_df.loc[add_stks, 'last_pr'] / (all_add_cnt + 1)
            wts_np[date_idx, self.hold_df.loc[add_stks, 'index']] += self.stk_pct
        return wts_np
    def NoAdd(self, date_idx, wts_np):
        return wts_np
    def long_N(self):
        return (self.hold_np[:, 1] - self.hold_np[:, 0]) / self.hold_np[:, 2]
    def short_N(self):
        return (self.hold_np[:, 0] - self.hold_np[:, 1]) / self.hold_np[:, 2]
    def long_pos(self, date_idx, wts_np, stks):
        wts_np[date_idx, self.hold_df.loc[stks, 'index']] += self.stk_pct
        return wts_np
    def short_pos(self, date_idx, wts_np, stks):
        wts_np[date_idx, self.hold_df.loc[stks, 'index']] -= self.stk_pct
        return wts_np
    def same_direction_pred(self, stk_df):
        pred_upper = self.rf_upper.predict(stk_df)
        pred_lower = self.rf_lower.predict(stk_df)
        all_stks = stk_df.index
        pos_stks = where(pred_upper > 0)[0]
        if len(pos_stks) > 0:
            sel_stks01 = intersect1d(pos_stks, where(pred_lower > 0)[0], assume_unique = True)
            sel_stks02 = where(pred_upper > -pred_lower)[0]
            return setdiff1d(all_stks, all_stks[union1d(sel_stks01, sel_stks02)])
        else:
            return all_stks
    def strict_same_direction_pred(self, stk_df):
        pred_upper = self.rf_upper.predict(stk_df)
        pred_lower = self.rf_lower.predict(stk_df)
        all_stks = stk_df.index
        pos_indices = where(pred_upper > 0)[0]
        min_cond = self.hold_np[:, 5] >= self.min_days
        cand_names = assets[self.hold_np[npnz(min_cond)[0], 6].astype('int16')]
        pos_names = all_stks[pos_indices]
        strict_names = intersect1d(pos_names, cand_names, assume_unique = True)
        if len(strict_names) > 0:
            sel_names01 = intersect1d(strict_names, all_stks[where(pred_lower > 0)[0]], assume_unique = True)
            sel_names02 = all_stks[where(pred_upper > -pred_lower)[0]]
            return setdiff1d(all_stks, union1d(sel_names01, sel_names02))
        else:
            return []
class Trader(nn.Module):
    def __init__(self, hidden_size: int, actor_act: nn, critic_act: nn):
        super(Trader, self).__init__()
        self.emb_sector = nn.Embedding(9, 7)
        self.emb_month = nn.Embedding(12, 7)
        self.emb_weekday = nn.Embedding(5, 3)
        self.actor_l1 = nn.Linear(32, hidden_size)#7+7+3+15=32
        self.actor_act2 = actor_act()
        self.actor_l3 = nn.Linear(hidden_size, 1)
        self.critic_l1 = nn.Linear(1, 4)
        self.critic_act2 = critic_act()
        self.critic_l3 = nn.Linear(4, 1)
        self.critic_tanh4 = nn.Tanh()
        for layer in (self.actor_l1, self.actor_l3, self.critic_l1, self.critic_l3):
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param, gain=1)
                else:
                    nn.init.constant_(param, 0)
        self.actor_model = nn.Sequential(self.actor_l1,
            nn.LayerNorm(hidden_size, elementwise_affine = True, bias = True), self.actor_act2, nn.AlphaDropout(p = 0.3), self.actor_l3)
        self.critic_model = nn.Sequential(self.critic_l1, self.critic_act2, self.critic_l3, self.critic_tanh4)
    def forward(self, state, liq_tensor):
        sector_ = self.emb_sector(state[:, -3].int())
        month_ = self.emb_month(state[:, -2].int())
        weekday_ = self.emb_weekday(state[:, -1].int())
        numerical_data = state[:, :-3]  # (N, cont_feat)
        x = tcat([numerical_data, sector_, weekday_, month_], dim=1)  # (N, 5+3+4+cont_feat)
        a3 = self.actor_model(x)
        a3_detach = a3.detach()
        a3_avg = a3_detach.mean().reshape(1)
        #print('a3 avg:', a3_avg, a3_avg.shape)
        critic = self.critic_model(a3_avg)
        #print('long critic:', critic, critic.shape)
        a4 = exp(log_softmax(a3.T.squeeze(0), dim = 0))
        actor = twhere(liq_tensor == 0, 0, a4)
        pure_actor = tdiv(actor, actor.sum())
        max_val = pure_actor.max()
        actor_detach = pure_actor.detach()
        if max_val == 0:
            print('Max probability = 0!!!!!!')
            print('a5:', a4)
            return 1, 1
        elif not tisfinite(actor_detach).all():
            print('a5 is not finite!!!!!!')
            print('a5:', a4)
            return 1, 1
        else:
            actor = twhere(liq_tensor == 0, -1, pure_actor)
            return actor, critic

def plot_score(result_dict: dict, lrs: list | dict, result_name: str, view = True):
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(24, 15))
    k_fraction, sub_fraction, blank, left_space, right_space, bottom_space, top_space = 0.5, 0.5, 0.01, 0.07, 0.02, 0.05, 0.05
    axk = fig.add_axes([left_space, 1 - k_fraction + blank, 1 -
                        left_space - right_space, k_fraction - blank - top_space])
    axv = fig.add_axes([left_space, bottom_space, 1 - left_space -
                        right_space, sub_fraction - bottom_space], sharex=axk)
    if type(lrs) == list:
        x_labels = range(len(lrs))
        axv.plot(x_labels, lrs, linewidth = 2.5, label = 'learning rate')
    else:
        for lr_key in lrs.keys():
            x_labels = range(len(lrs[lr_key]))
            axv.plot(x_labels, lrs[lr_key], linewidth = 2.5, label = lr_key)
    for k in result_dict.keys():
        axk.plot(x_labels, result_dict[k], linewidth = 2.5, label = k)

    plt.setp(axk.get_xticklabels(), visible = False)
    plt.setp(axv.get_xticklabels(), visible = True)
    plt.setp(axk.get_yticklabels(), visible = True)
    plt.setp(axv.get_yticklabels(), visible = True)
    axk.set_ylabel(ylabel = result_name, fontsize = 22)
    axk.tick_params(axis = 'y', labelsize = 22)
    axv.tick_params(axis = 'x', labelsize = 22)
    axv.tick_params(axis = 'y', labelsize = 22)
    axk.yaxis.set_major_locator(locator = ticker.AutoLocator())
    axk.yaxis.set_minor_locator(locator = ticker.AutoMinorLocator())
    axv.yaxis.set_major_locator(locator = ticker.AutoLocator())
    axv.yaxis.set_minor_locator(locator = ticker.AutoMinorLocator())
    axk.legend(framealpha = 0, markerscale = 0.8, fontsize = 22)
    axv.legend(framealpha = 0, markerscale = 0.8, fontsize = 22)
    axk.set_title(label = result_name.capitalize(), fontsize = 24)
    plot_fname = f'{folder}/{result_name}.png'
    plt.savefig(plot_fname, dpi = 280)
    if view:
        plt.show()
    plt.close('all')
def get_index_data(count):
    if 3 > count > 0:
        sleep(count * 5 + 15)
    elif count == 3:
        return 1
    hd = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36"}
    try:
        nq_url = f'https://api.nasdaq.com/api/quote/SPX/historical?assetclass=index&fromdate={prepare_date}&limit=9999&todate={end_date}&random=8'
        res = requests.get(nq_url, headers=hd)
        jdata = jloads(res.text)
        status = int(jdata['status']['rCode'])
        if status == 200:
            print('Index status code=200')
            data_dict = jdata['data']
            if data_dict is not None:
                table = data_dict['tradesTable']
                print('type of index table:', type(table))
                if type(table) == dict:
                    all_rows, new_data = table['rows'], {'time':[], 'spx':[]}
                    if all_rows is not None:
                        for row in all_rows:
                            single_date_lt, close_str = row['date'].split('/'), float(row['close'].replace(',', ''))
                            output_date = to_datetime(f'{single_date_lt[-1]}-{single_date_lt[0]}-{single_date_lt[1]}', format ='%Y-%m-%d')
                            new_data['time'].append(output_date)
                            new_data['spx'].append(close_str)
                        new_data = DataFrame(new_data).sort_values(by = ['time'], ascending = True, ignore_index = True)
                        return new_data
                    else:
                        print('All rows is none')
                        return get_index_data(count + 1)
                else:
                    print('Index: The table is empty')
                    return get_index_data(count + 1)
            else:
                print('Index Error occurs, bcode=', jdata['status']["bCodeMessage"][0]['code'], jdata['status']["bCodeMessage"][0]['errorMessage'])
                return get_index_data(count + 1)
        else:
            print(f'Index Error occurs, status code={status}\n', jdata['status']["bCodeMessage"][0]['errorMessage'])
            return get_index_data(count + 1)
    except Exception as err:
        print(err)
        return get_index_data(count + 1)
def next_trade_dt(last_trade_date):
    wd = last_trade_date.isoweekday()
    if wd == 5 or wd == 6:
        return Timestamp((last_trade_date + timedelta(8 % wd)).date())
    else:
        return Timestamp((last_trade_date + timedelta(1)).date())
def process_date(fst_date):
    fst_ind = date_idx_dict[fst_date]
    final_df = DataFrame(index=assets)
    for col_name in all_stk_cols:
        values = stat_dict[col_name].loc[fst_ind, :].values
        mean_val = values[isfinite(values)].mean()
        values = where(isfinite(values), values, mean_val)
        if col_name != 'logret10':
            try:
                values = yeojohnson(values)[0]
            except Exception as err:
                print(f'yeojohnson error: {err}, col_name: {col_name}, fst_date: {fst_date}')
        avg_val, std_val = values.mean(), values.std(ddof=1)
        values = (values - avg_val) / std_val
        final_df[col_name] = values
    cat_features = ['sector', 'month', 'weekday']
    final_df['sector'] = '0'
    sector_mapping = {idx: sector_dict[idx] for idx in final_df.index}
    final_df['sector'] = final_df.index.map(sector_mapping)
    fst_dt = Timestamp(fst_date).to_pydatetime()
    final_df['month'] = fst_dt.month
    final_df['weekday'] = fst_dt.isoweekday()
    tensor_df = final_df.copy(deep = True)
    tensor_df['sector'] = transform_sector(final_df['sector'])
    tensor_df['month'] -= 1
    tensor_df['weekday'] -= 1
    final_tensor = from_numpy(tensor_df.to_numpy().astype(float32))
    for col_name in cat_features:
        final_df[col_name] = final_df[col_name].astype('category')
    return fst_date, final_df, final_tensor

if __name__ == '__main__':
    cur_method = mp.get_start_method()
    print('Current start method:', cur_method)
    preferred_method = 'fork'
    if cur_method != preferred_method:
        mp.set_start_method(preferred_method, force = True)
        cur_method = mp.get_start_method()
        print('Update start method to:', cur_method)
    time_now = datetime.now().strftime('%Y%m%d-%H%M')
    env_dict = {'colab': ['/content/drive/MyDrive/stocks/',
                            '/content/drive/MyDrive/2016_2019/'], 'nb': ['D:/stocks/', 'D:/2016_2019/'],
                'kaggle': ['/kaggle/working/', '/kaggle/input/']}  # working存output data，input存input data
    print('env:', env)
    folder = f'{env_dict[env][0]}{time_now}'
    if not isdir(folder):
        mkdir(folder)
    st_time = time()
    start_date, end_date, prepare_date = '2006-01-01', '2025-05-16', '2005-03-18'
    dt_start = to_datetime(start_date)
    data = qndata.stocks.load_spx_data(min_date = prepare_date, max_date = end_date)
    spx_stk_lt = qndata.stocks.load_spx_list(min_date = prepare_date, max_date = end_date)
    sector_dict = {stk_dict['id']: stk_dict['sector'] if stk_dict['sector'] is not None else 'Unknown'
                  for stk_dict in spx_stk_lt}
    common_times = data.time.values
    last_trade_date = Timestamp(common_times[-1]).to_pydatetime()
    next_trade_date = next_trade_dt(last_trade_date)
    print('Next trade date:', next_trade_date)
    new_data = data.sel(time = last_trade_date)
    data = xr.concat([data, new_data], dim = 'time')
    update_times = data.time.values
    update_times[-1] = next_trade_date
    data = data.assign_coords(time = update_times)
    data = qnta.shift(data, periods = 1)

    spx_df = get_index_data(0)
    if type(spx_df) == int:
        spx_qnt = qndata.index.load_data(assets = ['SPX'], min_date = prepare_date).sel(asset='SPX')
        spx_qnt = spx_qnt.sel(time = common_times)
        spx_df = spx_qnt.to_pandas().to_frame()
        spx_df.reset_index(inplace = True)
        spx_df.rename(columns = {0: 'spx'}, inplace = True)
        print('Failed to get SPX data')
    else:
        print('Get SPX data successfully')
        spx_df = spx_df[spx_df['time'].isin(common_times)]
        print('SPX data length:', len(spx_df))
        print('SPX:\n', spx_df.tail(5))
    print('spx df head:\n', spx_df.head(5))

    vol, high, low = data.sel(field = 'vol'), data.sel(field = 'high'), data.sel(field = 'low')
    prices = data.sel(field = "close")
    print('After concat prices shape:', prices.to_pandas().shape)
    #print('Tail indices after shift:', prices.to_pandas().tail(5).index)
    assets = data.coords["asset"].values
    unique_vals, counts = unique(assets, return_counts = True)
    dup_vals = unique_vals[counts > 1]
    print('Duplicated values:', dup_vals)
    assets_cnt = assets.shape[0]
    print('assets count:', assets_cnt)
    assets_map_dict = {v: i for i, v in enumerate(assets)}
    pr_copy = prices.to_pandas()
    prices_df = pr_copy.copy(deep = True)
    prices_df.reset_index(inplace = True)
    prices_df = prices_df[prices_df['time'] >= dt_start]
    prices_df.reset_index(inplace = True, drop = True)
    pct_df = prices_df[assets].pct_change(fill_method = None, axis = 0)
    pct_df.fillna(value = 0, axis = 0, inplace = True)
    pct_np = pct_df.to_numpy()
    print('pct np shape:\n', pct_np.shape)
    prices_np = prices_df[assets].to_numpy()
    row_cnt = prices_np.shape[0]
    annual_coef = 252 ** 0.5
    initial_liq = data.sel(field="is_liquid")
    liquid_df = initial_liq.to_pandas()
    raw_liq_df = liquid_df.copy(deep = True)
    raw_liq_df = raw_liq_df[raw_liq_df.index >= dt_start]
    valid_liq = raw_liq_df.copy(deep = True)
    valid_liq.replace(0, NaN, inplace = True)
    raw_liq_df = raw_liq_df.fillna(value = 0, axis = 0)
    liquid_df.reset_index(inplace = True)
    spx_df['spx'] = spx_df['spx'].shift(1)
    index_ma = spx_df['spx'].rolling(200).mean()
    bias_df = DataFrame({'time': spx_df['time']})
    bias_df['bias'] = spx_df['spx'] / index_ma - 1
    bias_df = bias_df[bias_df['time'] >= dt_start]
    bias_df.reset_index(inplace = True, drop = True)
    bias_df['time'] = to_datetime(bias_df['time'])
    spx_df = spx_df[spx_df['time'] >= dt_start]
    spx_df.reset_index(inplace = True, drop = True)

    liquid_df = liquid_df[liquid_df['time'] >= dt_start]
    liquid_df.reset_index(inplace = True, drop = True)
    liquid_df[assets] = liquid_df[assets].fillna(value = 0, axis = 0)
    pr_last_idx = len(liquid_df) - 1
    returns = pr_copy.pct_change(fill_method = None)#開始算features，先不要對pr_copy reset_index
    returns.iloc[0, :] = 0
    returns.fillna(0, inplace = True, axis = 0)
    returns_arr = returns.to_numpy()
    vol_df = vol.to_pandas()
    vol_df.fillna(0, inplace = True, axis = 0)
    vol_arr = vol_df.to_numpy()
    stk_chg_inds = liquid_df.iloc[::10].index.values
    raw_stk_chg_inds = stk_chg_inds + 200
    all_dates = raw_liq_df.index.values
    stk_select_dates = all_dates[stk_chg_inds]
    monitor_dict = dict()
    chg_date = stk_select_dates[0]
    for next_chg_date in stk_select_dates[1:]:
        monitor_dict[chg_date] = all_dates[(all_dates > chg_date) & (all_dates < next_chg_date)]
        chg_date = next_chg_date
    monitor_dict[next_chg_date] = all_dates[all_dates > next_chg_date]
    valid_liq = DataFrame(array(valid_liq.loc[stk_select_dates, :]), index = stk_chg_inds, columns = assets)
    end_time01 = time()
    prepare_time = round((end_time01 - st_time) / 60, 3)
    print('Time to prepare data:', prepare_time, 'min')
    dbx_study_fname = f'/qnt_Adamax_lowpct_layernorm.pkl'
    local_study_fname = f'{folder}{dbx_study_fname}'
    dbx_params_fname = f'/params_Adamax_lowpct_layernorm.csv'
    local_params_fname = f'{folder}{dbx_params_fname}'
    dbx_df_fname = f'/qnt_Adamax_lowpct_layernorm.csv'
    local_df_fname = f'{folder}{dbx_df_fname}'
    dbx_wt_fname = f'/wt_Adamax_lowpct_layernorm.csv'
    local_wt_fname = f'{folder}{dbx_wt_fname}'
    dbx_long_model_fname = f'/long_model_Adamax_lowpct_layernorm.pt'
    local_long_model_fname = f'{folder}{dbx_long_model_fname}'
    dbx_short_model_fname = f'/short_model_Adamax_lowpct_layernorm.pt'
    local_short_model_fname = f'{folder}{dbx_short_model_fname}'
    dbx_gbdt_fname = f'/gbdt_final.joblib'
    local_gbdt_fname = f'{folder}{dbx_gbdt_fname}'
    dbx_dart_fname = f'/dart_final.joblib'
    local_dart_fname = f'{folder}{dbx_dart_fname}'
    dbx_rf_fname = f'/rf_final.joblib'
    local_rf_fname = f'{folder}{dbx_rf_fname}'
    dbx_upper_fname = f'/rf_upper.joblib'
    local_upper_fname = f'{folder}{dbx_upper_fname}'
    dbx_lower_fname = f'/rf_lower.joblib'
    local_lower_fname = f'{folder}{dbx_lower_fname}'
    dbx_encoder_fname = f'/sector_encoder.joblib'
    local_encoder_fname = f'{folder}{dbx_encoder_fname}'

    def train(model, opt, trade_type: int, hold_period: int, update_cnt: int, stk_cnt: int, return_dict: dict):
        #long_model, long_opt, 1, hold_period, update_cnt, return_dict
        set_num_threads(1)
        set_num_interop_threads(1)
        opt.zero_grad()
        gamma, values, logprobs, rewards, err = 0.97, [], [], [], False
        tmp_wts_df = raw_liq_df.copy(deep = True)
        tmp_wts_df.loc[:, assets] = 0
        tmp_wts_df.columns.names = ['asset']
        """
        ini_wt_dict = dict()
        for name, param in model.named_parameters():
            if 'weight' in name:
                ini_wt_dict[name] = param.data.clone()
        """
        true_train_dates = iter(train_dates[::hold_period])
        train_cnt = 0
        train_time_lt, stat_time_lt = [], []
        #tmp_wts = zeros((row_cnt, assets_cnt))
        for fst_date in true_train_dates:
            st_train_time = time()
            next_date = next(true_train_dates, fst_test_date)
            end_idx = date_indices[next_date]
            start_idx = date_indices[fst_date]
            #last_date = all_dates[end_idx - 1]
            #tmp_wts[start_idx: end_idx, :] = NaN
            stk_data = all_tensors_dict[fst_date]
            liq_tensor = liquid_tensor[start_idx, :]
            policy, predval = model(stk_data, liq_tensor)
            if type(policy) != int:
                logits = policy.view(-1)
                sel_indices = topk(logits, k = stk_cnt, largest = True, sorted = False)[1].numpy()
                prob = logits[sel_indices].sum()
                #tmp_wts[start_idx, sel_indices] = trade_type / select_cnt
            else:
                print(f'Model outputs wrong values during training')
                print('trade type:', trade_type)
                err = True
                break
            train_cnt += 1
            mon_liq = liquid_arr[start_idx: end_idx, sel_indices]
            #stat_wts = DataFrame(tmp_wts, index = prices_df['time'], columns = assets)
            stat_st_time = time()
            daily_pct = pct_np[start_idx: end_idx, sel_indices] * trade_type * mon_liq
            daily_ratio = 1 + daily_pct
            daily_val = cumprod(daily_ratio, axis = 0)
            daily_val[isnan(daily_val)] = 1
            max_val = maximum.accumulate(daily_val, axis = 0)
            mdd_val = (daily_val / max_val - 1).min(axis = 0)
            rew = nanmedian(daily_val[-1, :] - 1 + mdd_val) * annual_coef
            stat_end_time = time()
            stat_time_lt.append(stat_end_time - stat_st_time)
            values.append(predval)
            logprobs.append(prob)
            rewards.append(rew)
            if train_cnt % update_cnt == 0:
                Returns, ret_ = [], predval.detach()
                mod_rews = Tensor(rewards[-update_cnt:]).flip(dims = (0,)).view(-1)
                mod_probs = stack(logprobs[-update_cnt:]).flip(dims = (0,)).view(-1)
                mod_vals = stack(values[-update_cnt:]).flip(dims = (0,)).view(-1)
                discount_factors = gamma ** tarange(update_cnt)
                scaled_cumsum = tcumsum(mod_rews / discount_factors, dim=0)
                Returns = scaled_cumsum * discount_factors + ret_ * (gamma ** tarange(1, update_cnt + 1))
                actor_loss = -mod_probs * (Returns - mod_vals.detach())
                critic_loss = pow(mod_vals - Returns, 2)
                opt.zero_grad()
                loss = actor_loss.sum() + 0.1 * critic_loss.sum()
                #print('loss:', loss)
                loss.backward()
                #print('backprop:', mon_cnt)
                clip_grad_norm_(model.parameters(), max_norm = 5, norm_type = 2.0, error_if_nonfinite = True, foreach = True)
                opt.step()
            end_train_time = time()
            train_time_lt.append(end_train_time - st_train_time)
        """
        for name, param in model.named_parameters():
            if 'weight' in name:
                if equal(ini_wt_dict[name], param):
                    print('Name:', name)
                    print('ini wt:', ini_wt_dict[name].shape, ini_wt_dict[name])
                    print('current wt:', param.shape, param)
                    raise ValueError('The params did not update!!!')
        """
        #med_time = round(nanmedian(train_time_lt) / 60, 3)
        #print(f'-----Median train time: {med_time} min-----')
        if not err:
            result = {'model': model, 'opt': opt, 'time': train_time_lt, 'stat_time': stat_time_lt}
            return_dict[trade_type] = result
        else:
            return_dict[trade_type] = 1
    def process(total_epochs, hold_period, update_cnt, stk_pct, params_dict):
        #total_epochs, hold_period, update_cnt, stk_pct, params_dict
        long_model, short_model = params_dict['long']['model'], params_dict['short']['model']
        long_opt, short_opt = params_dict['long']['opt'], params_dict['short']['opt']
        long_cnt, short_cnt = params_dict['long']['cnt'], params_dict['short']['cnt']
        long_lrs, short_lrs, test_scores, max_score, stop_cnt, patience, stop, stk_cnt = [], [], [], NINF, 0, total_epochs // 3, False, long_cnt + short_cnt
        for epoch in range(total_epochs):
            if not stop:
                #print(f'Start epoch {epoch}')
                ctx = mp.get_context(preferred_method)
                with ctx.Manager() as manager:
                    return_dict = manager.dict()
                    #print('Create return dict')
                    processes = [
                        ctx.Process(
                            target = train,
                            args = (long_model, long_opt, 1, hold_period, update_cnt, stk_cnt, return_dict)
                        ),
                        ctx.Process(
                            target = train,
                            args = (short_model, short_opt, -1, hold_period, update_cnt, stk_cnt, return_dict)
                        )
                    ]
                    #print('Create process list')
                    for p in processes:
                        p.start()
                    #print('Start processes')
                    for p in processes:
                        p.join()
                    #print('Join processes')
                    long_result = return_dict.get(1, 1)
                    short_result = return_dict.get(-1, 1)
                    #print('long err sign:', long_err, 'short err sign:', short_err)
                    if type(long_result) != int and type(short_result) != int:
                        long_model, long_opt, long_time_lt, long_stat_time_lt = long_result['model'], long_result['opt'], long_result['time'], long_result['stat_time']
                        short_model, short_opt, short_time_lt, short_stat_time_lt = short_result['model'], short_result['opt'], short_result['time'], short_result['stat_time']
                        params_dict['long']['model'] = long_model
                        params_dict['short']['model'] = short_model
                        st_test_time = time()
                        sharpe, arr, mdd, volat, calmar, stats_df, wts_df = test_model(stk_pct, params_dict)
                        end_test_time = time()
                        test_model_time = round((end_test_time - st_test_time) / 60, 5)
                        trial_obj.test_time_lt.append(test_model_time)
                        trial_obj.train_time_lt.extend(long_time_lt)
                        trial_obj.train_time_lt.extend(short_time_lt)
                        trial_obj.stat_time_lt.extend(long_stat_time_lt)
                        trial_obj.stat_time_lt.extend(short_stat_time_lt)
                        if type(wts_df) != int:
                            median_sharpe = nanmedian(stats_df['sharpe_ratio'])
                            test_score = (sharpe + median_sharpe) / 2 - mdd
                            long_model.train()
                            short_model.train()
                            long_lrs.append(long_opt.param_groups[0]['lr'])
                            short_lrs.append(short_opt.param_groups[0]['lr'])
                            test_scores.append(test_score)
                            if test_score > trial_obj.score:
                                trial_obj.score = test_score
                                trial_obj.sharpe, trial_obj.arr, trial_obj.mdd, trial_obj.vol, trial_obj.calmar = sharpe, arr, mdd, volat, calmar
                                print('Updated the best score:', test_score, 'Sharpe:', sharpe, 'ARR:', arr, 'MDD:', mdd, 'Volat:', volat, 'Calmar:', calmar)
                                print('Median sharpe:', round(median_sharpe, 4))
                                trial_obj.long_state_dict = deepcopy(long_model.state_dict())
                                trial_obj.short_state_dict = deepcopy(short_model.state_dict())
                                trial_obj.long_lrs = long_lrs
                                trial_obj.short_lrs = short_lrs
                                trial_obj.test_scores = test_scores
                                trial_obj.median_sharpe = median_sharpe
                                trial_obj.stats_df = stats_df.copy(deep = True)
                                trial_obj.wts_df = wts_df.copy(deep = True)
                                trial_obj.stk_cnt = stk_cnt
                            if test_score > max_score:
                                max_score = test_score
                                stop_cnt = 0
                            elif test_score <= max_score - 0.15:
                                stop_cnt += 1
                                if stop_cnt == patience:
                                    print(f'Early stopping at epoch {epoch}')
                                    stop = True
                                    break
                        else:
                            print(f'Model outputs wrong values during testing at epoch: {epoch}')
                            stop = True
                            break
                    else:
                        print(f'Model outputs wrong values during training at epoch: {epoch}')
                        stop = True
                        break
        if len(test_scores) > 0:
            return max(test_scores)
        else:
            raise TrialPruned

    with open(f'{env_dict[env][1]}dropbox/dropbox.json') as jf:
        jdata = jload(jf)
    dbx = Dropbox(app_key=jdata['app_key'],
                    app_secret=jdata['app_secret'],
                    oauth2_refresh_token=jdata['oauth2_refresh_token'])
    try:
        dbx.users_get_current_account()
    except AuthError as err:
        exit(
            "ERROR: Invalid access token; try re-generating an access token from the app console on the web.")
    if download_data(dbx, local_gbdt_fname, dbx_gbdt_fname) != 1:
        model_data = load(local_gbdt_fname)
        long_gbdt = model_data['long']
        short_gbdt = model_data['short']
    else:
        raise ValueError('The data of the gbdt model does not exist')

    if download_data(dbx, local_dart_fname, dbx_dart_fname) != 1:
        model_data = load(local_dart_fname)
        long_dart = model_data['long']
        short_dart = model_data['short']
    else:
        raise ValueError('The data of the dart model does not exist')

    if download_data(dbx, local_rf_fname, dbx_rf_fname) != 1:
        model_data = load(local_rf_fname)
        long_rf = model_data['long']
        short_rf = model_data['short']
    else:
        raise ValueError('The data of the rf model does not exist')

    if download_data(dbx, local_upper_fname, dbx_upper_fname) != 1:
        model_data = load(local_upper_fname)
        long_upper = model_data['long_up']
        short_upper = model_data['short_up']
    else:
        raise ValueError('The data of the upper model does not exist')
    if download_data(dbx, local_lower_fname, dbx_lower_fname) != 1:
        model_data = load(local_lower_fname)
        long_lower = model_data['long_low']
        short_lower = model_data['short_low']
    else:
        raise ValueError('The data of the lower model does not exist')

    model_dict = {'long': {'gbdt': long_gbdt, 'dart': long_dart, 'rf': long_rf, 'upper': long_upper, 'lower': long_lower},
                    'short': {'gbdt': short_gbdt, 'dart': short_dart, 'rf': short_rf, 'upper': short_upper, 'lower': short_lower}}
    if download_data(dbx, local_encoder_fname, dbx_encoder_fname) != 1:
        label_encoder = load(local_encoder_fname)
    else:
        raise ValueError('The data of the label encoder does not exist')
    st_time02 = time()
    date_idx_dict = dict(zip(stk_select_dates, stk_chg_inds))
    stat_dict = get_features(returns_arr, vol_arr)
    all_stk_cols = array(list(stat_dict.keys()))
    total_dates_cnt = len(stk_select_dates)
    split_index = total_dates_cnt * 3 // 5 + 1#減少train dates數量以便加速，測試完後需要改回total_dates_cnt // 2 + 1
    print('Split index:', split_index)
    train_dates = stk_select_dates[1: split_index]
    test_dates = stk_select_dates[split_index:]
    date_indices = {date_: npnz(all_dates == date_)[0][0] for date_ in all_dates}
    last_train_date = train_dates[-1]
    fst_test_date = test_dates[0]
    fst_select_date = stk_select_dates[0]
    all_features_dict, all_tensors_dict = dict(), dict()
    results = Parallel(n_jobs=-1)(delayed(process_date)(fst_date) for fst_date in stk_select_dates)
    for fst_date, final_df, final_tensor in results:
        all_features_dict[fst_date] = final_df
        all_tensors_dict[fst_date] = final_tensor
    liquid_arr = liquid_df[assets].to_numpy()
    nonliq_np_dict = dict()
    for fst_date in stk_select_dates:
        start_idx = where(all_dates == fst_date)[0][0]
        liq_arr = liquid_arr[start_idx, :]
        nonliq_np_dict[fst_date] = where(liq_arr != 1)[0]

    liquid_tensor = from_numpy(liquid_arr)
    end_time02 = time()
    feature_time = round((end_time02 - st_time02) / 60, 3)
    print('Time to generate features:', feature_time, 'min')
    print('Total time:', prepare_time + feature_time, 'min')
    batch, input_col_len, func_lt = len(assets), 18, ['ELU', 'SELU', 'GELU', 'CELU', 'Sigmoid', 'SiLU', 'Mish', 'Softplus', 'Softshrink', 'Softsign', 'Tanhshrink', 'Tanh']
    trial_obj, stop_model_dict = BestTrial(), dict()
    print('input stocks col len:', input_col_len)

    download_val = download_data(
        dbx, local_study_fname, dbx_study_fname)
    if download_val != 1:
        study = load(download_val)
        best_trial_num, trial_cnt = study.best_trial.number, 45
        trial_best_score = study.best_trial.value
        if download_data(dbx, local_df_fname, dbx_df_fname) != 1:
            trial_obj.stats_df = read_csv(local_df_fname, parse_dates=['time'])
        else:
            raise ValueError('The dataframe of the best params does not exist')
        if download_data(dbx, local_params_fname, dbx_params_fname) != 1:
            best_params = read_csv(local_params_fname).to_dict()
            trial_obj.sharpe = best_params['sharpe'][0]
            trial_obj.arr = best_params['arr'][0]
            trial_obj.mdd = best_params['mdd'][0]
            trial_obj.vol = best_params['vol'][0]
            trial_obj.calmar = best_params['calmar'][0]
            trial_obj.score = best_params['score'][0]
            trial_obj.median_sharpe = best_params['median_sharpe'][0]
            print('Previous best params:')
            for k, v in best_params.items():
                print(f'{k}: {v[0]}')
        else:
            raise ValueError('The data of the best params does not exist')
        if download_data(dbx, local_wt_fname, dbx_wt_fname) != 1:
            trial_obj.wts_df = read_csv(local_wt_fname, parse_dates=['time'])
        else:
            raise ValueError('The data of the wt does not exist')
    else:
        study = create_study(direction = "maximize", pruner = MedianPruner())
        best_trial_num, best_params, trial_cnt = NaN, {
            'name': time_now, 'score': 0}, 45
    study.optimize(objective, n_trials = trial_cnt, timeout = None, show_progress_bar = False)
    pruned_trials = study.get_trials(
        deepcopy = False, states = [TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy = False, states = [TrialState.COMPLETE])
    print("Number of finished trials:", len(study.trials))
    print("Number of pruned trials:", len(pruned_trials))
    print("Number of complete trials:", len(complete_trials))
    print("The training score of the best trial:", study.best_trial.value)
    trial_best_df, best_wt, trial_best_score = trial_obj.stats_df, trial_obj.wts_df, trial_obj.score
    if study.best_trial.number == best_trial_num:
        same_best_trial = True
        print('The best trial is the same as before.')
        print('The best score:', trial_best_score)
    else:
        same_best_trial = False
        best_params['name'] = time_now
        best_params['score'] = trial_obj.score
        best_params['sharpe'] = trial_obj.sharpe
        best_params['arr'] = trial_obj.arr
        best_params['vol'] = trial_obj.vol
        best_params['mdd'] = trial_obj.mdd
        best_params['calmar'] = trial_obj.calmar
        best_params['median_sharpe'] = trial_obj.median_sharpe
        print('Best params:')
        for key, value in study.best_trial.params.items():
            best_params[key] = value
            print(f'{key}: {value}')
        print('Updated the best trial')
        print('Updated the best score', trial_best_score)
        DataFrame(best_params, index = [0]).to_csv(local_params_fname, index=False)
        upload_data(dbx, local_params_fname, dbx_params_fname)
        trial_best_df = trial_best_df.merge(right = spx_df, how = 'inner', on = 'time', sort = True)
        best_wt.reset_index(inplace = True)
        best_wt['sum'] = best_wt[assets].sum(axis = 1)
        sum_df = DataFrame({'time': best_wt['time'], 'sum': best_wt['sum']})
        trial_best_df = trial_best_df.merge(right = sum_df, how = 'inner', on = 'time', sort = True)
        #trial_best_df.dropna(subset = ['time'], inplace = True, how = 'any')
        trial_best_df.to_csv(local_df_fname, index = False)
        upload_data(dbx, local_df_fname, dbx_df_fname)
    dump(study, local_study_fname)
    upload_data(dbx, local_study_fname, dbx_study_fname)
    strat = 'Strategy'
    compare_df = DataFrame({'date': trial_best_df['time'], 'SPX': 100 * trial_best_df['spx'] / trial_best_df.loc[0, 'spx'] - 100,
    strat: 100 * trial_best_df['equity'] / trial_best_df.loc[0, 'equity'] - 100,
    'SPX DD': 100 * trial_best_df['spx'] / trial_best_df['spx'].cummax() - 100,
    f'{strat} DD': 100 * trial_best_df['equity'] / trial_best_df['equity'].cummax() - 100,
    'pos': trial_best_df['sum'], 'indicator': bias_df.loc[: len(trial_best_df) - 1, 'bias']})
    y_groups = compare_df.groupby([Grouper(key = 'date', freq = 'YE')])
    for y in y_groups:
        plot_year = y[0][0].to_pydatetime().year#pandas 2.2.2版本在timestamp的物件外包了一個tuple，所以要多加一個[0]
        compare_return(y[1], 'SPX', strat, plot_year)
    compare_return(compare_df, 'SPX', strat, 'All')
    print('Best Score:', trial_best_score)
    print('Best Sharpe:', trial_obj.sharpe)
    print('Best ARR:', trial_obj.arr)
    print('Best MDD:', trial_obj.mdd)
    print('Best VOL:', trial_obj.vol)
    print('Best Calmar:', trial_obj.calmar)
    print('Best Median Sharpe:', trial_obj.median_sharpe)
    if not same_best_trial:
        longmodel_dict = {'model_state_dict': trial_obj.long_state_dict, 'name': time_now}
        save({**longmodel_dict, **best_params}, local_long_model_fname)
        shortmodel_dict = {'model_state_dict': trial_obj.short_state_dict, 'name': time_now}
        save({**shortmodel_dict, **best_params}, local_short_model_fname)
        upload_data(dbx, local_long_model_fname, dbx_long_model_fname)
        upload_data(dbx, local_short_model_fname, dbx_short_model_fname)
        print('Updated the best score')
        plot_score({'Test Score': trial_obj.test_scores}, {'long lr': trial_obj.long_lrs, 'short lr': trial_obj.short_lrs}, 'score')
        best_wt['num'] = count_nonzero(best_wt[assets], axis = 1)
        best_wt.to_csv(local_wt_fname, index=False)
        upload_data(dbx, local_wt_fname, dbx_wt_fname)
    else:
        print('The best score is the same as before.')
    plot_param_importances(study)
    plot_optimization_history(study)
    med_test_time = nanmedian(trial_obj.test_time_lt)
    med_train_time = round(nanmedian(trial_obj.train_time_lt) / 60, 3)
    med_stat_time = round(nanmedian(trial_obj.stat_time_lt) / 60, 5)
    print('--------Median test model time:', med_test_time, 'min--------')
    print('--------Median train model time:', med_train_time, 'min--------')
    print('--------Median stat time:', med_stat_time, 'min--------')
    #print('---Correlation---')
    #qnstats.print_correlation(wts_xr, data)
    #contest_type = "stocks_s&p500"
    #wts_xr = qnout.clean(wts_xr, data, contest_type)
    #qnout.check(wts_xr, data, contest_type, check_correlation = True)
    #qnout.write(wts_xr)
