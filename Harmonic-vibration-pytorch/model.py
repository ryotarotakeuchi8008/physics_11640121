import torch
import torch.nn as nn

# 入力系列・出力系列の次元
INPUT_FEATURES = 1
LAYER1_NEURONS = 4
LAYER2_NEURONS = 4
LAYER3_NEURONS = 4
LAYER4_NEURONS = 4
LAYER5_NEURONS = 4
LAYER6_NEURONS = 4
OUTPUT_NEURONS = 1

activation1 = torch.nn.Tanh()
activation2 = torch.nn.Tanh()
activation3 = torch.nn.Tanh()
activation4 = torch.nn.Tanh()
activation5 = torch.nn.Tanh()
activation6 = torch.nn.Tanh()
activation_out = torch.nn.Identity()

# RNNの各種パラメーター
input_size = 1
hidden_size = 16


# 中間層1つの通常ニューラルネットワーク
class NeuralNetwork0(nn.Module):
    def __init__(self):
        super(NeuralNetwork0, self).__init__()

        self.layer1 = nn.Linear(
            INPUT_FEATURES, LAYER1_NEURONS
        )
        self.layer2 = nn.Linear(
            LAYER1_NEURONS, LAYER2_NEURONS
        )
        self.layer_out = nn.Linear(
            LAYER2_NEURONS, OUTPUT_NEURONS
        )

    # foward propagation
    def forward(self, x):
        x = activation1(self.layer1(x))
        x = activation2(self.layer2(x))
        x = activation_out(self.layer_out(x))
        return x


# 中間層2つの通常ニューラルネットワーク
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(
            INPUT_FEATURES, LAYER1_NEURONS
        )
        self.layer2 = nn.Linear(
            LAYER1_NEURONS, LAYER2_NEURONS
        )
        self.layer3 = nn.Linear(
            LAYER2_NEURONS, LAYER3_NEURONS
        )
        self.layer4 = nn.Linear(
            LAYER3_NEURONS, LAYER4_NEURONS
        )
        self.layer_out = nn.Linear(
            LAYER4_NEURONS, OUTPUT_NEURONS
        )

    # foward propagation
    def forward(self, x):
        x = activation1(self.layer1(x))
        x = activation2(self.layer2(x))
        x = activation3(self.layer3(x))
        x = activation4(self.layer4(x))
        x = activation_out(self.layer_out(x))
        return x


# 中間層6つの通常ニューラルネットワーク
class NeuralNetwork2(nn.Module):
    def __init__(self):
        super(NeuralNetwork2, self).__init__()

        self.layer1 = nn.Linear(
            INPUT_FEATURES, LAYER1_NEURONS
        )
        self.layer2 = nn.Linear(
            LAYER1_NEURONS, LAYER2_NEURONS
        )
        self.layer3 = nn.Linear(
            LAYER2_NEURONS, LAYER3_NEURONS
        )
        self.layer4 = nn.Linear(
            LAYER3_NEURONS, LAYER4_NEURONS
        )
        self.layer5 = nn.Linear(
            LAYER4_NEURONS, LAYER5_NEURONS
        )
        self.layer6 = nn.Linear(
            LAYER5_NEURONS, LAYER6_NEURONS
        )
        self.layer_out = nn.Linear(
            LAYER6_NEURONS, OUTPUT_NEURONS
        )

    # foward propagation
    def forward(self, x):
        x = activation1(self.layer1(x))
        x = activation2(self.layer2(x))
        x = activation3(self.layer3(x))
        x = activation4(self.layer4(x))
        x = activation5(self.layer5(x))
        x = activation6(self.layer6(x))
        x = activation_out(self.layer_out(x))
        return x


# RNNを用いる場合
class RNNNeuralNetwork(nn.Module):
    def __init__(self):
        super(RNNNeuralNetwork, self).__init__()

        self.rnn = torch.nn.RNN(input_size, hidden_size)

        self.fc = torch.nn.Linear(hidden_size, 1)

    # foward propagation
    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)

        output = self.fc(output[:, -1])
        return output, hidden
