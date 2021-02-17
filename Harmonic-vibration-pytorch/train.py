import torch
import torch.nn as nn
from utils import load_dataset, plot_history
from utils2 import load_dataset2
from dataProductor import data_productor
from model import RNNNeuralNetwork, NeuralNetwork, NeuralNetwork2, NeuralNetwork0
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# hyperParameter


# 各パラメーター
input_size = 1  # 入力系列の次元
output_size = 1  # 出力系列の次元
batch_size = 20
hidden_size = 12  # 隠れ状態の次元
num_layers = 1
num_epochs = 10
maxlen = 300
model_path = 'models/model.h5'

# optimizer周り
LEARNING_RATE = 0.03
REGULARIZATION = 0.03


def train_step(train_t, train_label, model, criterion, optimizer):
    # 訓練モードに設定
    model.train()

    output = model(train_t)

    optimizer.zero_grad()

    loss = criterion(output, train_label[-1, :])

    # back propagation
    loss.backward(retain_graph=True)
    # 勾配を使って各parameterを更新
    optimizer.step()

    return loss


def valid_step(valid_t, valid_label, model, criterion):
    # 評価モード
    model.eval()

    # RNNに入れるときだけ3階テンソルにしないといけない
    valid_t = valid_t.reshape(-1, batch_size, input_size)

    # 初期の隠れ状態テンソル (RNNに必要)
    hidden = torch.zeros(num_layers, batch_size, hidden_size)

    output, hidden = model(valid_t, hidden)

    loss = criterion(output, valid_label[-1, :])

    return loss


def init_parameters(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)  # 全重みをランダム値で初期化
        layer.bias.data.fill_(0.0)  # バイアスを0で初期化


def main():
    data_productor()  # データの作成

    # Data loading. ミニバッチ学習用
    loader_train, loader_valid = load_dataset(
        'data/data.csv', batch_size)

    # クラスの実行 initメソッドが呼ばれ、モデルの各層のインスタンスをインスタンス変数として定義
    # model = NeuralNetwork0()
    # model = NeuralNetwork()
    # model = NeuralNetwork2()
    model = RNNNeuralNetwork()

    model.apply(init_parameters)  # 学習前に各パラメーターを初期化

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

        # サブセットごとに精度をミニバッチvalidate
        for valid_t, valid_label in loader_valid:
            loss = valid_step(valid_t, valid_label, model, criterion)

            running_loss_valid += loss.item()

        # エポックの平均の損失値(train)
        running_loss_train /= len(loader_train)
        losses_train.append(running_loss_train)

        # エポックの平均の損失値(valid)

        running_loss_valid /= len(loader_valid)
        losses_valid.append(running_loss_valid)

        print("epoch: {}, loss_train: {},loss_valid:{}, acc: {}".format(epoch, round(running_loss_train, 2),
                                                                        round(running_loss_valid, 2), running_acc))

    # ここからloss値のグラフ表示

    # 1. figureを生成する
    fig = plt.figure()

    # 2. 生成したfigureにaxesを生成、配置する
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # 3. axesに描画データを設定する
    X_train = range(num_epochs)
    Y_train = losses_train
    X_valid = range(num_epochs)
    Y_valid = losses_valid

    ax1.plot(X_train, Y_train, label="loss(Training data)")

    ax1.set_title("train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid()

    ax2.set_title("valid loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.grid()

    ax3.set_title("inference output")
    ax3.set_xlabel("t")
    ax3.set_ylabel("x")
    ax3.grid()

    # 4. 表示する
    plt.show()

    print("finished train & validate")

    print(model.state_dict())

    # 学習で取得したモデルの情報(各層の各parameter(wやb))をdict型で取得
    params = model.state_dict()

    # ↑で取得したモデルの情報(各層の各parameter(wやb))をファイルに保存
    torch.save(params, "checkpoints/model.prm")


if __name__ == '__main__':
    main()
