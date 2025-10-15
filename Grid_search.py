import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import os
import joblib
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from itertools import product
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

tf.config.set_visible_devices([], 'GPU')


def calculate_kge(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / (np.std(y_true) + 1e-10)
    beta = np.mean(y_pred) / (np.mean(y_true) + 1e-10)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def calculate_nse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - numerator / denominator


def create_lagged_features(df, lag=1):
    df_lagged = df.copy()
    for col in ['v_in_interp']:
        for i in range(1, lag + 1):
            df_lagged[f'{col}_lag{i}'] = df_lagged[col].shift(i)
    return df_lagged


def create_time_windows(X, y, time_array, window_size=5):
    X_windows, y_windows, t_windows = [], [], []
    for i in range(len(X) - window_size):
        X_windows.append(X[i:i + window_size])
        y_windows.append(y[i + window_size - 1])
        t_windows.append(time_array[i + window_size - 1])
    return np.array(X_windows), np.array(y_windows), np.array(t_windows)


def train_single_fold(args):
    (X_train, y_train, X_val, y_val, val_idx,
     y_scaler, fold_num, repeat, input_shape, lstm_units, epochs) = args

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    model = Sequential([
        LSTM(lstm_units, activation='tanh', input_shape=input_shape),
        Dropout(0.5),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=32, verbose=0)

    y_val_pred_scaled = model.predict(X_val, verbose=0)
    y_val_true = y_scaler.inverse_transform(y_val)
    y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled)

    val_kge = calculate_kge(y_val_true.flatten(), y_val_pred.flatten())
    val_nse = calculate_nse(y_val_true.flatten(), y_val_pred.flatten())

    y_train_pred_scaled = model.predict(X_train, verbose=0)
    y_train_true = y_scaler.inverse_transform(y_train)
    y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled)

    train_kge = calculate_kge(y_train_true.flatten(), y_train_pred.flatten())
    train_nse = calculate_nse(y_train_true.flatten(), y_train_pred.flatten())

    return {
        'fold': fold_num,
        'repeat': repeat,
        'val_idx': val_idx,
        'y_val_pred': y_val_pred,
        'y_val_true': y_val_true,
        'train_kge': train_kge,
        'train_nse': train_nse,
        'val_kge': val_kge,
        'val_nse': val_nse,
        'train_loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1]
    }


def train_with_params(params_combo, X_scaled, y_scaled, time_array, features,
                      feature_name, n_folds, n_repeats, y_scaler, max_workers):
    window_size, epochs, lstm_units = params_combo

    X_windowed, y_windowed, t_windowed = create_time_windows(
        X_scaled, y_scaled, time_array, window_size)

    X_windowed = X_windowed.reshape(
        (X_windowed.shape[0], X_windowed.shape[1], X_windowed.shape[2]))

    all_repeat_predictions = []
    all_experiments = []

    for repeat in range(1, n_repeats + 1):
        kfold = KFold(n_splits=n_folds, shuffle=False)
        fold_results = []

        input_shape = (X_windowed.shape[1], X_windowed.shape[2])
        fold_args = []

        fold_num = 1
        for train_idx, val_idx in kfold.split(X_windowed):
            X_train, X_val = X_windowed[train_idx], X_windowed[val_idx]
            y_train, y_val = y_windowed[train_idx], y_windowed[val_idx]

            fold_args.append((X_train, y_train, X_val, y_val, val_idx,
                              y_scaler, fold_num, repeat, input_shape,
                              lstm_units, epochs))
            fold_num += 1

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(train_single_fold, args) for args in fold_args]

            for future in as_completed(futures):
                result = future.result()
                fold_results.append(result)

                experiment_record = {
                    'Feature_Combination': feature_name,
                    'Window_Size': window_size,
                    'Epochs': epochs,
                    'LSTM_Units': lstm_units,
                    'Batch_Size': 32,
                    'Repeat': result['repeat'],
                    'Fold': result['fold'],
                    'Train_KGE': result['train_kge'],
                    'Train_NSE': result['train_nse'],
                    'Val_KGE': result['val_kge'],
                    'Val_NSE': result['val_nse'],
                    'Train_Loss': result['train_loss'],
                    'Val_Loss': result['val_loss']
                }
                all_experiments.append(experiment_record)

        fold_results.sort(key=lambda x: x['fold'])

        stitched_predictions = np.zeros((len(y_windowed), 1))
        stitched_true = np.zeros((len(y_windowed), 1))

        for fold_result in fold_results:
            val_idx = fold_result['val_idx']
            stitched_predictions[val_idx] = fold_result['y_val_pred']
            stitched_true[val_idx] = fold_result['y_val_true']

        all_repeat_predictions.append(stitched_predictions)

    all_repeat_predictions = np.array(all_repeat_predictions)
    median_predictions = np.median(all_repeat_predictions, axis=0)

    median_kge = calculate_kge(stitched_true.flatten(), median_predictions.flatten())
    median_nse = calculate_nse(stitched_true.flatten(), median_predictions.flatten())

    return {
        'params': params_combo,
        'window_size': window_size,
        'epochs': epochs,
        'lstm_units': lstm_units,
        'kge': median_kge,
        'nse': median_nse,
        'predictions': median_predictions,
        'time_stamps': t_windowed,
        'y_true': stitched_true,
        'experiments': all_experiments
    }


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 8})

    data_path = "path/to/your/data.csv"  # Path to CSV file containing training data with columns: time, v_in_interp, v_out, wse1, wse2, slope_cm_per_km
    df = pd.read_csv(data_path)
    df['v_in_interp'] /= 1000
    df['v_out'] /= 1000
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    df['v_in_interp'] = df['v_in_interp'].interpolate(method='linear', limit_direction='both')

    df = create_lagged_features(df, lag=1)
    df.dropna(inplace=True)

    feature_combinations = [
        ['v_in_interp', 'v_in_interp_lag1', 'wse1', 'wse2', 'slope_cm_per_km'],
        ['v_in_interp', 'v_in_interp_lag1', 'slope_cm_per_km'],
        ['v_in_interp', 'v_in_interp_lag1', 'wse1', 'wse2'],
        ['v_in_interp', 'v_in_interp_lag1']
    ]

    feature_names = [
        'Qout (Qin, WSE, WSS)',
        'Qout (Qin, WSS)',
        'Qout (Qin, WSE)',
        'Qout (Qin)'
    ]

    param_grid = {
        'window_size': [4, 6, 8],
        'epochs': [50, 100, 150],
        'lstm_units': [16, 32, 64]
    }

    target = 'v_out'
    n_folds = 5
    n_repeats = 5
    max_workers = min(32, os.cpu_count())

    save_dir = "path/to/save/directory"
    csv_dir = "path/to/csv/directory"
    fig_dir = "path/to/figure/directory"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    param_combinations = list(product(
        param_grid['window_size'],
        param_grid['epochs'],
        param_grid['lstm_units']
    ))

    all_results = {}
    all_experiments_list = []

    for feat_idx, features in enumerate(feature_combinations):

        X = df[features].values
        y = df[target].values.reshape(-1, 1)

        x_scaler = StandardScaler()
        X_scaled = x_scaler.fit_transform(X)
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y)

        feature_results = []

        for param_idx, params in enumerate(param_combinations):
            result = train_with_params(
                params, X_scaled, y_scaled, df['time'].values,
                features, feature_names[feat_idx],
                n_folds, n_repeats, y_scaler, max_workers
            )

            feature_results.append(result)
            all_experiments_list.extend(result['experiments'])

        best_result = max(feature_results, key=lambda x: x['kge'])

        all_results[feature_names[feat_idx]] = {
            'best_params': best_result['params'],
            'best_kge': best_result['kge'],
            'best_nse': best_result['nse'],
            'all_results': feature_results,
            'predictions': best_result['predictions'],
            'time_stamps': best_result['time_stamps'],
            'y_true': best_result['y_true']
        }

    experiments_df = pd.DataFrame(all_experiments_list)
    csv_path = os.path.join(csv_dir, 'all_grid_search_results.csv')
    experiments_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    best_params_summary = []
    for feat_name, result in all_results.items():
        best_params_summary.append({
            'Feature_Combination': feat_name,
            'Best_Window_Size': result['best_params'][0],
            'Best_Epochs': result['best_params'][1],
            'Best_LSTM_Units': result['best_params'][2],
            'Batch_Size': 32,
            'Best_KGE': result['best_kge'],
            'Best_NSE': result['best_nse']
        })

    best_params_df = pd.DataFrame(best_params_summary)
    best_params_path = os.path.join(csv_dir, 'best_parameters.csv')
    best_params_df.to_csv(best_params_path, index=False, encoding='utf-8-sig')

    best_params_json = {}
    for feat_name, result in all_results.items():
        best_params_json[feat_name] = {
            'window_size': int(result['best_params'][0]),
            'epochs': int(result['best_params'][1]),
            'lstm_units': int(result['best_params'][2]),
            'batch_size': 32,
            'kge': float(result['best_kge']),
            'nse': float(result['best_nse'])
        }

    json_path = os.path.join(save_dir, 'best_parameters.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(best_params_json, f, indent=4, ensure_ascii=False)

    plt.figure(figsize=(9 / 2.54, 5 / 2.54))

    first_feat = feature_names[0]
    time_stamps = all_results[first_feat]['time_stamps']
    y_true = all_results[first_feat]['y_true']

    plt.plot(time_stamps, y_true, label='True outflow', linewidth=1, color='black')

    line_styles = ['dashdot', 'dashed', 'dotted', '-']
    colors = ['black', 'black', 'black', 'gray']

    for i, feat_name in enumerate(feature_names):
        result = all_results[feat_name]
        plt.plot(result['time_stamps'], result['predictions'],
                 label=f'{feat_name} (KGE={result["best_kge"]:.2f}, NSE={result["best_nse"]:.2f})',
                 linewidth=1, color=colors[i], linestyle=line_styles[i])

    plt.ylabel('Release (10³ m³/s)', labelpad=0)
    plt.xlabel('Date', labelpad=0)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.legend(frameon=False, fontsize=6)
    plt.gca().tick_params(axis='both', direction='in', which='both', size=1.5)
    plt.subplots_adjust(left=0.08, right=0.97, top=0.96, bottom=0.13)
    out_fp = os.path.join(fig_dir, ".png")
    plt.savefig(out_fp, dpi=800)

    print("\n" + "=" * 80)
    print("Grid Search Results Summary")
    print("=" * 80)
    for feat_name, result in all_results.items():
        params = result['best_params']
        print(f"{feat_name}:")
        print(f"  Best params: window_size={params[0]}, epochs={params[1]}, lstm_units={params[2]}")
        print(f"  KGE={result['best_kge']:.3f}, NSE={result['best_nse']:.3f}")

    print("\n" + "=" * 80)
    print("Output Files")
    print("=" * 80)
    print(f"Prediction plot: {out_fp}")
    print(f"All results CSV: {csv_path}")
    print(f"Best parameters CSV: {best_params_path}")
    print(f"Best parameters JSON: {json_path}")
    print("=" * 80)

    try:
        plt.show()
    except:
        pass