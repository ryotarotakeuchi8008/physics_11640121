import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


# CSVからデータ読み取り・前処理用の関数
def load_dataset(filename, batch_size):
    df = pd.read_csv(filename, names=('t', 'x', "y", "v_x ", "v_y"),
                     )

    # 読み込んだdfデータ(DataFrame型)から中身をdf.valuesの形でnumpyの配列(numpy.ndarray型)として取り出す(そのままスライシングしたり、行列の演算に使える)
    myarray = df.values

    # 　下記で学習時用を0~9000行・validate時用を9001行~10000行として取得するのでここで行をランダムにすることを忘れずに(np.random.shuffleは破壊的メソッド)
    # ここないと軌道の0~9000行だけを学習して軌道の9001行~10000行目を予測できなくなる
    np.random.shuffle(myarray)


    # numpyの多次元配列のスライシング機能
    t_numpy_array = myarray[:, 0]
    x_numpy_array = myarray[:, 1]
    y_numpy_array = myarray[:, 2]
    v_x_numpy_array = myarray[:, 3]
    v_y_numpy_array = myarray[:, 4]


    # それぞれ列ベクトルに変換
    t_numpy_array = t_numpy_array.reshape(len(t_numpy_array), 1)
    x_numpy_array = x_numpy_array.reshape(len(x_numpy_array), 1)
    y_numpy_array = y_numpy_array.reshape(len(y_numpy_array), 1)
    v_x_numpy_array = v_x_numpy_array.reshape(len(v_x_numpy_array), 1)
    v_y_numpy_array = v_y_numpy_array.reshape(len(v_y_numpy_array), 1)

    # 入力テンソル
    input_sequence = t_numpy_array

    # 列ベクトルを連結して教師ラベルのテンソルとする
    label_matrix = np.hstack((x_numpy_array, y_numpy_array, v_x_numpy_array, v_y_numpy_array))

    # 入力テンソルと正解テンソルをそれぞれtrain用とvalidate用に分ける(numpyの多次元配列のスライシング機能)
    input_sequence_train = input_sequence[0:9000, :]
    input_sequence_validate = input_sequence[9000:, :]

    label_matrix_train = label_matrix[0:9000, :]
    label_matrix_validate = label_matrix[9000:, :]

    # NumPy多次元配列からtensor型に変換し、データ型はfloatに変換する
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
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size)

    return loader_train, loader_valid


def load_dataset2(filename):
    df = pd.read_csv(filename, names=('t', 'x', "y", "v_x ", "v_y"),
                     )

    # 読み込んだdfデータ(DataFrame型)から中身をnumpyの配列(numpy.ndarray型)として取り出す(そのままスライシングしたり、行列の演算に使える)
    myarray = df.values

    # ここないと軌道の0~9000行だけを学習して軌道の9001行~10000行目を予測できなくなる
    np.random.shuffle(myarray)

    # numpyの多次元配列のスライシング機能
    t_numpy_array = myarray[:, 0]
    x_numpy_array = myarray[:, 1]
    y_numpy_array = myarray[:, 2]
    v_x_numpy_array = myarray[:, 3]
    v_y_numpy_array = myarray[:, 4]

    # それぞれ列ベクトルに変換
    t_numpy_array = t_numpy_array.reshape(len(t_numpy_array), 1)
    x_numpy_array = x_numpy_array.reshape(len(x_numpy_array), 1)
    y_numpy_array = y_numpy_array.reshape(len(y_numpy_array), 1)
    v_x_numpy_array = v_x_numpy_array.reshape(len(v_x_numpy_array), 1)
    v_y_numpy_array = v_y_numpy_array.reshape(len(v_y_numpy_array), 1)

    # 入力テンソル
    input_sequence = t_numpy_array

    # 列ベクトルを連結して教師ラベルのテンソルとする
    label_matrix = np.hstack((x_numpy_array, y_numpy_array, v_x_numpy_array, v_y_numpy_array))

    # 入力テンソルと正解テンソルをそれぞれtrain用とvalidate用に分ける(numpyの多次元配列のスライシング機能)
    input_sequence_train = input_sequence[0:9000, :]
    input_sequence_validate = input_sequence[9000:, :]

    label_matrix_train = label_matrix[0:9000, :]
    label_matrix_validate = label_matrix[9000:, :]

    # NumPy多次元配列からtensor型に変換し、データ型はfloatに変換する
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
    loader_train2 = DataLoader(dataset_train, batch_size=9000, shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=1000)

    return loader_train2, loader_valid


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
