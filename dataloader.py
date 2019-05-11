import torch
import torch.utils.data as data
import numpy as np
from random import randrange
from util import Util


# データローダの定義（データを1つずつ読み込むもの）
class ReberGrammarSet(data.Dataset):
    # 初期化メソッド
    def __init__(self, filename, length):
        super(ReberGrammarSet, self).__init__()
        self.length = length
        with open(filename, 'r') as file:
            self.texts = ['@' * self.length + line.rstrip() + '@'
                          for line in file]
        # self.texts = self.texts[:10000]

    # indexを指定してデータを取得するメソッド
    def __getitem__(self, index):
        position = randrange(self.length, len(self.texts[index]))
        input_text = self.texts[index][position - self.length: position]
        output_character = self.texts[index][position]
        input_tensor = torch.tensor(np.stack([Util.get_one_hot_vector(c)
                                              for c in input_text])).float()
        output_tensor = torch.tensor(Util.get_index(output_character),
                                     dtype=torch.long)
        return input_tensor, output_tensor

    # データセットのサイズを取得するメソッド
    def __len__(self):
        return len(self.texts)
