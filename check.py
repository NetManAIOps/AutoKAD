import os
import torch
import numpy as np
import random
import json
import pandas as pd

from detector.HW import train_and_predict as hw_train_and_predict
from detector.LSTM import train_and_predict as lstm_train_and_predict
from detector.VAE import train_and_predict as donut_train_and_predict

from utils.preprocess import minmax_scale

import matplotlib.pyplot as plt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_and_predict(train_df, test_df, params):
    algo_name = params['name']

    train_and_predict = None
    if algo_name == 'holt_winter':
        train_and_predict = hw_train_and_predict
    elif algo_name == 'LSTM':
        train_and_predict = lstm_train_and_predict
    elif algo_name == 'Donut':
        train_and_predict = donut_train_and_predict

    try:
        predicted = train_and_predict(train_df['value'], test_df['value'], params['params'])
    except Exception as e:
        print("params conflict!")
        print(e)

    # Make sure the prediction is one-dimension array
    predicted = predicted.reshape(-1)
    # replace nan with 0
    predicted[np.isnan(predicted)] = .0

    return predicted


def load_data(kpi):
    train_df = pd.read_csv(f"data/tzs/train/{kpi}.csv")
    test_df = pd.read_csv(f"data/tzs/test/{kpi}.csv")

    train_df['value'] = minmax_scale(train_df['value'])
    test_df['value'] = minmax_scale(test_df['value'])

    return train_df, test_df


def plot(ground_truth: np.ndarray, predicted: np.ndarray, save_path=None, labels=None, anomaly_score=None, figsize=(20, 5)):
    plot_num = 2
    if anomaly_score is not None:
        plot_num += 1

    plt.figure(figsize=(figsize[0], figsize[1] * plot_num))

    x = np.arange(0, len(ground_truth))

    # Test fig
    plt.subplot(plot_num, 1, 1)
    y = ground_truth
    plt.plot(x, y)
    plt.title("Test")

    if labels is not None:
        anomaly_pos = labels != 0
        plt.plot(x[anomaly_pos], y[anomaly_pos], marker='o', linestyle='None', color='r', markersize=2)

    # predicted fig
    plt.subplot(plot_num, 1, 2)
    y = predicted
    plt.plot(x, y)
    plt.title("Prediction")


    # Anomaly score
    plt.subplot(plot_num, 1, 3)
    y = anomaly_score
    plt.plot(x, y)
    plt.title("Anomaly Score")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


if __name__ == '__main__':
    setup_seed(329)

    kpi = "a8c06b47-cc41-3738-9110-12df0ee4c721"
    os.makedirs(f"plot/{kpi}",  exist_ok=True)

    check_file = "trials/msenf_tpe_100.json"
    trials = json.load(open(check_file, 'r'))

    train_df, test_df = load_data(kpi)

    cnt = 1

    for trial in trials:
        print(json.dumps(trial, indent=4, sort_keys=True))

        algo_name = trial['name']
        predicted = train_and_predict(train_df, test_df, trial)
        anomaly_score = abs(predicted - test_df['value'].to_numpy())




        # plot(ground_truth=test_df['value'].to_numpy(),
        #      predicted=predicted,
        #      labels=test_df['label'].to_numpy(),
        #      anomaly_score=anomaly_score,
        #      save_path=f"plot/{cnt}.png")

        cnt += 1




