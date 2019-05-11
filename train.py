import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import random
from dataloader import ReberGrammarSet
from model import Predictor
from util import Util
from fastprogress import master_bar, progress_bar
import time


# コマンドライン引数の受け取り
parser = argparse.ArgumentParser(description='Classifier demo with PyTorch')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
parser.add_argument('--length', type=int, default=5, help='length of input sequence')
parser.add_argument('--hidden_size', type=int, default=10, help='hidden vector size')
option = parser.parse_args()

# 実行ごとに結果が変わらないように乱数シードを固定
random.seed(1)
torch.manual_seed(option.seed)

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# データローダの用意
train_set = ReberGrammarSet('reber_train_2.4M.txt', option.length)
train_data_loader = DataLoader(dataset=train_set, batch_size=option.batchSize,
                               shuffle=True)
test_set = ReberGrammarSet('reber_test_1M.txt', option.length)
test_data_loader = DataLoader(dataset=test_set, batch_size=1)

# 分類器と誤差関数の用意
predictor = Predictor(Util.num_characters, option.hidden_size).to(device)
criterion = nn.CrossEntropyLoss()

# Adamオプティマイザを用意
optimizer = optim.Adam(predictor.parameters(), lr=option.lr)

# ネットワーク構造の表示
print(predictor)


def train():
    mb = master_bar(range(option.nEpochs))

    for epoch in mb:
        start_time = time.time()

        # 学習
        predictor.train()  # 学習モードに変更
        avg_loss = 0

        # イテレーション
        for sequence, answer in progress_bar(train_data_loader, parent=mb):
            sequence.to(device)
            answer.to(device)
            predicted = predictor.forward(sequence)  # 前向き計算（ラベル予測）
            loss = criterion(predicted, answer)  # クロスエントロピー誤差の計算

            optimizer.zero_grad()  # オプティマイザが保持する勾配を初期化
            loss.backward()  # 後ろ向き計算（誤差逆伝播で勾配計算）
            optimizer.step()  # オプティマイザでパラメータを修正
            avg_loss += loss.item() / len(train_data_loader)

        # 今エポックのロスの平均値を出力
        print('Average Loss: {:.04f}'.format(avg_loss))

        # テスト
        predictor.eval()  # 評価モードに変更
        with torch.no_grad():  # 無駄な勾配計算を行わないようにする
            accuracy = 0

            # テストデータのイテレーションを回す
            for sequence, answer in test_data_loader:
                sequence.to(device)
                answer.to(device)
                predicted = predictor.forward(sequence)  # 前向き計算（ラベル予測）
                predicted_label = torch.argmax(predicted)  # 確率が最大となるラベルを予測ラベルとする
                if predicted_label.item() == answer.item():
                    accuracy += 1 / len(test_data_loader)

            # 全体の正解率を表示
            print('accuracy: {:.04f}'.format(accuracy))

        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            mb.write(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f} - '
                     f'accuracy: {accuracy:.4f} - time: {elapsed:.0f}s')

    # 学習済みモデルを保存
    torch.save({'state_dict': predictor.state_dict()}, 'checkpoint.pth')


if __name__ == '__main__':
    train()
