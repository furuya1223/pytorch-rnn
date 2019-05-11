from torch import nn


class Predictor(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super(Predictor, self).__init__()
        # モデル定義
        self.rnn = nn.RNN(num_classes, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    # 前向き計算
    def forward(self, x):
        x, h = self.rnn(x)
        x = x[:, -1]
        x = self.linear(x)
        return x
