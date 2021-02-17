import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


def load_dataset(filename, batch_size):
    df = pd.read_csv(filename, names=('t', 'x'),
                     )

    # 読み込んだdfデータ(DataFrame型)から中身をdf.valuesの形でnumpyの配列(numpy.ndarray型)として取り出す(そのままスライシングしたり、行列の演算に使える)
    myarray = df.values

    # numpyの多次元配列のスライシング機能(列でスライス)
    t_numpy_array = myarray[:, 0]
    x_numpy_array = myarray[:, 1]


    # それぞれ列ベクトルに変換
    t_numpy_array = t_numpy_array.reshape(len(t_numpy_array), 1)
    x_numpy_array = x_numpy_array.reshape(len(x_numpy_array), 1)


    # 入力テンソル
    input_sequence = t_numpy_array

    # 列ベクトルを連結して教師ラベルのテンソルとする
    label_matrix = x_numpy_array

    # 入力テンソルと正解テンソルをそれぞれtrain用とvalidate用に分ける(numpyの多次元配列のスライシング機能)
    input_sequence_train = input_sequence[0:56000, :]
    input_sequence_validate = input_sequence[0:21, :]


    label_matrix_train = label_matrix[0:56000, :]
    label_matrix_validate = label_matrix[0:21, :]

    # NumPy多次元配列からtensor型に変換し、データ型はfloatに変換する (pytorchで扱うときはtensor型)
    # train用
    input_train = torch.from_numpy(input_sequence_train).float()
    label_train = torch.from_numpy(label_matrix_train).float()

    # validate用
    input_validate = torch.from_numpy(input_sequence_validate).float()
    label_validate = torch.from_numpy(label_matrix_validate).float()


    # 入力データと教師ラベルを1つのデータセットとしてまとめる
    dataset_train = TensorDataset(input_train, label_train)
    dataset_valid = TensorDataset(input_validate, label_validate)

    # データセットの各データをbatch_sizeごとのサブセット(==ミニバッチ)に分けてロード,返り値はサブセットを各要素とするiterable object
    loader_train = DataLoader(dataset_train, batch_size=batch_size)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size)

    return loader_train, loader_valid


def plot_history(history):
    # Setting
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    # Plotting loss
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()

    # Plotting accuracy
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
