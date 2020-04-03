import json
import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import ast

class QUESTDataset(data.Dataset):
    def __init__(self, data_path):
        super(QUESTDataset, self).__init__()

        self.data = pd.read_csv(data_path)
        self.train = True if "scores" in self.data.columns else False

    def __getitem__(self, index):
        row = self.data.iloc[index]
        question = torch.as_tensor(ast.literal_eval(row["question"]))
        answer = torch.as_tensor(ast.literal_eval(row["answer"]))
        if self.train:
            scores = torch.as_tensor(ast.literal_eval(row["scores"]))
            return {'question': question, 'answer': answer, 'scores': scores}
        else:
            return {'question': question, 'answer': answer}

    def __len__(self):
        return len(self.data)
