import torch
import torch.nn as nn
import pandas as pd
from dataProductor import data_productor
from model import RNNNeuralNetwork
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation

input_size = 1  # 入力系列の次元
batch_size = 20
num_layers = 1  # RNNの層数
hidden_size = 16  # 隠れ状態の次元


def main():
    df = pd.read_csv('data/data.csv', names=('t', 'x', 'y', 'vx', 'vy')
                     )
    # 読み込んだdfデータ(DataFrame型)から中身をdf.valuesの形でnumpyの配列(numpy.ndarray型)として取り出す(そのままスライシングしたり、行列の演算に使える)
    myarray = df.values

    x_numpy_array_train = myarray[0:12000, 1:3]  # numpyの多次元配列のスライシング機能(列でスライス)

    count = len(x_numpy_array_train) - batch_size

    # 入力データの作成
    input_data = [x_numpy_array_train[idx:idx + batch_size] for idx in range(count)]
    num_batch = len(input_data)  # バッチの数
    input_data = torch.tensor(input_data, dtype=torch.float)  # list型からtensor型
    input_data = input_data.reshape(num_batch, batch_size, -1)  # RNNに入れる用に3階テンソルに変形

    # モデルの用意
    model = RNNNeuralNetwork()
    param_load = torch.load("./checkpoints/model2.prm")
    model.load_state_dict(param_load)

    output_history = []

    for i in range(10):
        # 隠れ状態の初期化

        hidden = torch.zeros(num_layers, batch_size, hidden_size)
        output, hidden = model(input_data, hidden)

        # 既存のx(0)~x(t)と出力した次の時刻x(t+1)を連結して次の入力とする
        output = output.reshape(num_batch, 1, 2)  # 既存のinput_dataに連結するためにoutputを3階テンソルに整形
        input_data = torch.cat([input_data[:, 1:, :2], output], dim=1).detach()

        output_numpy = output.to('cpu').detach().numpy()[:, 0, :2]
        output_history.append(output_numpy)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    t0 = [round(i * 0.07, 4) for i in range(0, 400, 1)]
    t1 = [round(i * 0.07, 4) for i in range(20, 400, 1)]
    t2 = [round(i * 0.07, 4) for i in range(21, 401, 1)]
    t3 = [round(i * 0.07, 4) for i in range(22, 402, 1)]
    t4 = [round(i * 0.07, 4) for i in range(23, 403, 1)]
    t5 = [round(i * 0.07, 4) for i in range(24, 404, 1)]


    c1, c2, c3, c4, c5, c6 = "black", "red", "green", "orange", "purple", "pink"

    ax1.plot(x_numpy_array_train[:, 0], x_numpy_array_train[:, 1], label="input_data", color=c1)
    ax1.plot(output_history[0][:, 0], output_history[0][:, 1], label="inference_data1", color=c2)
    ax1.plot(output_history[1][:, 0], output_history[1][:, 1], label="inference_data2", color=c3)
    ax1.plot(output_history[2][:, 0], output_history[2][:, 1], label="inference_data3", color=c4)
    # ax1.plot(t3, output_history[2], label="inference_data3", color=c4)
    # ax1.plot(t4, output_history[3], label="inference_data4", color=c5)

    ax1.set_title("x(t)-y(t)")
    ax1.set_xlabel("x(t)")
    ax1.set_ylabel("y(t)")

    ax1.grid()

    plt.legend(loc=(0.8, 0.8))
    plt.show()


if __name__ == '__main__':
    main()
