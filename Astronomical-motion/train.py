import torch
import torch.nn as nn
from utils import load_dataset, plot_history, load_dataset2
from dataProductor import data_productor
from model import NeuralNetwork
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# hyperParameter


# 各パラメーター
batch_size = 100
num_epochs = 50
maxlen = 300
model_path = 'models/model.h5'

# optimizer周り
LEARNING_RATE = 0.03
REGULARIZATION = 0.03


def train_step(train_t, train_label, model, criterion, optimizer):
    # 訓練モードに設定
    model.train()

    # モデルインスタンスの実行 forward propagation
    pred = model(train_t)

    optimizer.zero_grad()


    loss = criterion(pred, train_label)
    # back propagation
    loss.backward()
    # 勾配を使って各parameterを更新
    optimizer.step()

    return loss


def valid_step(valid_t, valid_label, model, criterion):
    # 評価モード
    model.eval()
    pred = model(valid_t)

    loss = criterion(pred, valid_label)

    return loss


def init_parameters(layer):
    if type(layer) == nn.Linear:
        # 全重みをランダム値で初期化
        nn.init.xavier_uniform_(layer.weight)
        # バイアスを0で初期化
        layer.bias.data.fill_(0.0)


def main():

    # データの作成
    data_productor()


    # Data loading. 返り値はサブセット(==batch_size個のデータセット)を各要素とするiterable object
    loader_train, loader_valid = load_dataset(
        'data/data.csv', batch_size)


    # クラスの実行 initメソッドが呼ばれ、モデルの各層のインスタンスをインスタンス変数として定義
    model = NeuralNetwork()

    # 学習前に各パラメーターを初期化
    model.apply(init_parameters)

    # 損失関数
    criterion = nn.MSELoss()

    # optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=REGULARIZATION
    )

    # エポックを繰り返した時の1エポックあたりの平均loss値の履歴
    losses_train = []
    losses_valid = []

    # エポック数だけiterate
    for epoch in range(num_epochs):
        # 1エポックごとの累計損失値(train)
        running_loss_train = 0.0
        running_acc = 0.0

        # 1エポックごとの累計損失値(valid)
        running_loss_valid = 0.0

        # サブセットごとにミニバッチ学習
        for train_t, train_label in loader_train:
            loss = train_step(train_t, train_label, model, criterion, optimizer)
            running_loss_train += loss.item()


        # サブセットごとに精度をvalidate
        for valid_t, valid_label in loader_valid:
            loss = valid_step(valid_t, valid_label, model, criterion)
            running_loss_valid += loss.item()



        # 1エポックの平均の損失値(train)
        running_loss_train /= len(loader_train)
        losses_train.append(running_loss_train)

        # 1エポックの平均の損失値(valid)
        running_loss_valid /= len(loader_valid)
        losses_valid.append(running_loss_valid)

        print("epoch: {}, loss_train: {},loss_valid:{}, acc: {}".format(epoch, round(running_loss_train, 2),
                                                                        round(running_loss_valid, 2), running_acc))

    # ここからloss値のグラフ表示
    # 1. figureを生成する
    fig = plt.figure()

    # 2. 生成したfigureにaxesを生成、配置する
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)

    # 3. axesに描画データを設定する
    X_train = range(num_epochs)
    Y_train = losses_train
    X_valid = range(num_epochs)
    Y_valid = losses_valid

    ax1.plot(X_train, Y_train, label="loss(Training data)")
    ax2.plot(X_valid, Y_valid, label="loss(Training data)")

    ax1.set_title("train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid()

    ax2.set_title("valid loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.grid()

    # 4. 表示する
    # plt.show()

    print("finished train & validate")

    print(model.state_dict())

    # 学習で取得したモデルの情報(各層の各parameter(wやb))をdict型で取得
    params = model.state_dict()

    # ↑で取得したモデルの情報(各層の各parameter(wやb))をファイルに保存
    torch.save(params, "checkpoints/model.prm")

    # 実際に出力して確認
    loader_train2, loader_valid2 = load_dataset2(
        'data/data.csv')
    for train_t, train_label in loader_train2:
        pred2 = model(train_t)
        import pdb;
        pdb.set_trace()

        #
        ax = fig.add_subplot(1, 4, 3)
        ims = []

        rx_list = pred2[:, 0].tolist()
        ry_list = pred2[:, 1].tolist()
        vx_list = pred2[:, 2].tolist()
        vy_list = pred2[:, 3].tolist()

        gx = []
        gy = []
        for i in range(len(pred2)):

            if i % 20 == 0:
                gx.append(rx_list[i])
                gy.append(ry_list[i])

                # ➀これまでのx値・y値を全てプロット → 線の軌跡に
                im_line = ax.plot(gx, gy, 'b')

                # ➁現在のx値・y値を点としてプロット
                im_point = ax.plot(rx_list[i], ry_list[i], marker='.', color='b', markersize=10)

                # # ③時間もプロット
                # im_time = ax.text(0.0, 0.1, 'time = {0:5.2f}'.format(t))

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

