import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


def load_dataset2(filename, batch_size):
    df = pd.read_csv(filename, names=('t', 'x', 'y', 'vx', 'vy'),
                     )

    # 読み込んだdfデータ(DataFrame型)から中身をdf.valuesの形でnumpyの配列(numpy.ndarray型)として取り出す(そのままスライシングしたり、行列の演算に使える)
    myarray = df.values

    # train用データ
    r_numpy_array_train = myarray[0:2300, 1:3]  # numpyの多次元配列のスライシング機能(列でスライス)

    # r_list_train = r_numpy_array_train.tolist()  # numpy型からlist型
    count = len(r_numpy_array_train) - batch_size  # 22980

    # 入力データの作成
    data_train = [r_numpy_array_train[idx:idx + batch_size, :] for idx in range(count)]
    num_batch = len(data_train)  # バッチの数

    data_train = torch.tensor(data_train, dtype=torch.float)  # list型からtensor型
    data_train = data_train.reshape(num_batch, batch_size, -1)  # RNNに入れる用に3階テンソルに変形
    # 正解ラベルの作成
    labels_train = [r_numpy_array_train[idx + batch_size] for idx in range(count)]
    labels_train = torch.tensor(labels_train, dtype=torch.float)
    labels_train = labels_train.reshape(num_batch, -1)

    # validate用データの作成
    x_numpy_array_validate = myarray[0:1000, 1:3]  # numpyの多次元配列のスライシング機能(列でスライス)

    count = len(x_numpy_array_validate) - batch_size
    # 入力データの作成
    data_validate = [x_numpy_array_validate[idx:idx + batch_size] for idx in range(count)]
    num_batch = len(data_validate)  # バッチの数
    data_validate = torch.tensor(data_validate, dtype=torch.float)  # list型からtensor型
    data_validate = data_validate.reshape(num_batch, batch_size, -1)  # RNNに入れる用に3階テンソルに変形
    # 正解ラベルの作成
    labels_validate = [x_numpy_array_validate[idx + batch_size] for idx in range(count)]
    labels_validate = torch.tensor(labels_validate, dtype=torch.float)
    labels_validate = labels_validate.reshape(num_batch, -1)

    return data_train, labels_train, data_validate, labels_validate
