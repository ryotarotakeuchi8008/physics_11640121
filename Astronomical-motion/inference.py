import numpy as np
import torch
from model import NeuralNetwork
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv


def round_int(x):
    return round(x, 3)


def main():
    # ルンゲ=クッタ法の幅dtと漸化式の繰り返し数  ( 無次元化した２次元微分方程式 t∗ = nt/dt )
    dt = 0.001
    nt = 10000
    t = 0
    t_list = []
    ims = []

    for i in range(nt):
        t += dt
        t_list.append(round(t, 3))

    # pythonのlist→Numpy ndarray
    t_list_np = np.array(t_list).reshape(len(t_list), 1)

    # Numpy ndarray→Pytorchのtensor
    input_tensor = torch.from_numpy(t_list_np).float()

    # インスタンスとしてモデルの定義
    model = NeuralNetwork()

    # 学習で取得し、ファイルに保存したモデルの情報(各層の各parameter(wやb))をloadしてモデルに適用→学習済みのモデルを適用
    param_load = torch.load("checkpoints/model.prm")
    model.load_state_dict(param_load)

    # モデルインスタンスの実行 forward propagation
    pred = model(input_tensor)

    with open('./data/data.csv2', 'a') as f:
        writer = csv.writer(f)
        for i in range(10000):
            # 各要素を丸める(小数点)
            t = input_tensor[i]
            x = pred[i, 0]
            y = pred[i, 1]
            writer.writerow([t, x, y])

    # グラフ描画 matplotlib
    fig = plt.figure()
    #
    ax = fig.add_subplot(1, 1, 1)

    rx_list = pred[:, 0].tolist()
    ry_list = pred[:, 1].tolist()
    vx_list = pred[:, 2].tolist()
    vy_list = pred[:, 3].tolist()

    for i in range(len(pred)):

        if i % 20 == 0:
            # ➀これまでのx値・y値を全てプロット → 線の軌跡に
            im_line = ax.plot(rx_list, ry_list, 'b')

            # ➁現在のx値・y値を点としてプロット
            im_point = ax.plot(rx_list[i], ry_list[i], marker='.', color='b', markersize=10)

            # ➀~③のaxsオブジェクトをまとめて配列に格納
            ims.append(im_line + im_point)

    # 漸化式をnt回繰り返し終わったらグラフ描画

    # 太陽の位置を座標指定してプロット
    ax.plot(0.0, 0.0, marker='.', color='r', markersize=10)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', 'datalim')

    # figureオブジェクトとaxsオブジェクトの配列を渡してアニメーション
    anm = animation.ArtistAnimation(fig, ims, interval=50)
    anm.save('animation.gif', writer='pillow')
    plt.show()


if __name__ == '__main__':
    main()
