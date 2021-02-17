import torch
import torch.nn as nn
import pandas as pd
from dataProductor import data_productor
from model import RNNNeuralNetwork, NeuralNetwork, NeuralNetwork2, NeuralNetwork0
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation

input_size = 1  # 入力系列の次元
batch_size = 20
num_layers = 1  # RNNの層数
hidden_size = 16  # 隠れ状態の次元


def main():
    df = pd.read_csv('data/data.csv', names=('t', 'x'),
                     )
    # 読み込んだdfデータ(DataFrame型)から中身をdf.valuesの形でnumpyの配列(numpy.ndarray型)として取り出す(そのままスライシングしたり、行列の演算に使える)
    myarray = df.values

    # test用データ
    # x_numpy_array_train = myarray[0:batch_size, 1]  # numpyの多次元配列のスライシング機能(列でスライス) 0~20の位置x
    # x_list = x_numpy_array_train.tolist()  # numpy型からlist型
    #
    # input_data = torch.tensor(x_list, dtype=torch.float)  # list型からtensor型
    # input_data = input_data.reshape(1, batch_size, -1)  # RNNに入れる用に3階テンソルに変形 バッチの個数×バッチサイズ×input_size

    x_numpy_array_train = myarray[0:400, 1]  # numpyの多次元配列のスライシング機能(列でスライス)

    x_list = x_numpy_array_train.tolist()  # numpy型からlist型
    count = len(x_list) - batch_size

    # 入力データの作成
    input_data = [x_list[idx:idx + batch_size] for idx in range(count)]
    num_batch = len(input_data)  # バッチの数
    input_data = torch.tensor(input_data, dtype=torch.float)  # list型からtensor型
    input_data = input_data.reshape(num_batch, batch_size, -1)  # RNNに入れる用に3階テンソルに変形

    # モデルの用意
    model = RNNNeuralNetwork()
    param_load = torch.load("./checkpoints/model2.prm")
    model.load_state_dict(param_load)

    output_history = []

    for i in range(100):
        # 隠れ状態の初期化

        hidden = torch.zeros(num_layers, batch_size, hidden_size)
        output, hidden = model(input_data, hidden)

        # 既存のx(0)~x(t)と出力した次の時刻x(t+1)を連結して次の入力とする
        output = output.reshape(num_batch, 1, -1)  # 既存のinput_dataに連結するためにoutputを3階テンソルに整形
        input_data = torch.cat([input_data[:, 1:, :], output], dim=1).detach()

        output_numpy = output.to('cpu').detach().numpy()[:, 0]
        output_history.append(output_numpy)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    t0 = [round(i * 0.07, 4) for i in range(0, 400, 1)]
    t1 = [round(i * 0.07, 4) for i in range(20, 400, 1)]
    t2 = [round(i * 0.07, 4) for i in range(21, 401, 1)]
    t3 = [round(i * 0.07, 4) for i in range(22, 402, 1)]
    t4 = [round(i * 0.07, 4) for i in range(23, 403, 1)]
    t5 = [round(i * 0.07, 4) for i in range(24, 404, 1)]

    # t3 = [i for i in range(20, 401, 1)]
    # t4 = [i for i in range(21, 402, 1)]

    # import pdb;
    # pdb.set_trace()

    c1, c2, c3, c4, c5, c6 = "black", "red", "green", "orange", "purple", "pink"

    ax1.plot(t0, x_list, label="input_data", color=c1)
    ax1.plot(t1, output_history[0], label="inference_data1", color=c2)
    # ax1.plot(t2, output_history[1], label="inference_data2", color=c3)
    # ax1.plot(t3, output_history[2], label="inference_data3", color=c4)
    # ax1.plot(t4, output_history[3], label="inference_data4", color=c5)

    ax1.set_title("t-x(t)")
    ax1.set_xlabel("t")
    ax1.set_ylabel("x(t)")
    # ax1.set_xlim([0, 500])
    # ax1.set_ylim([0, 3])
    ax1.grid()

    # 凡例(==ax.plot()のlabel)をまとめて表示
    plt.legend(loc=(0.8, 0.8))
    plt.show()


if __name__ == '__main__':
    main()
