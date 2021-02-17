import torch
import torch.nn as nn

# 入力系列・出力系列の次元
INPUT_FEATURES = 1
LAYER1_NEURONS = 6
LAYER2_NEURONS = 6
OUTPUT_NEURONS = 4

activation1 = torch.nn.Tanh()
activation2 = torch.nn.Tanh()
activation_out = torch.nn.Identity()

# RNNの各種パラメーター
input_size = 2
hidden_size = 16


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

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


# RNNを用いる場合
class RNNNeuralNetwork(nn.Module):
    def __init__(self):
        super(RNNNeuralNetwork, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, 2)

    # foward propagation
    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output[:, -1, :])
        return output, hidden
