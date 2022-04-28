import torch.nn

from utils import *


class Head(torch.nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Head, self).__init__()
        self.lstm = torch.nn.LSTM(in_dim, 100, batch_first=True,bidirectional=True,num_layers=3)
        self.relu = torch.nn.ReLU()
        self.l_norm = torch.nn.LayerNorm(200)
        self.lin1 = torch.nn.Linear(200, 100)
        self.drop = torch.nn.Dropout(0.2)
        self.lin2 = torch.nn.Linear(100, out_dim)
        self.softmax = torch.nn.LogSoftmax(dim=-1)



    def forward(self,x):
        x = self.lstm(x)[0]
        x = self.relu(x)
        x = self.l_norm(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.lin2(x)
        x = self.softmax(x)
        return x


class PunctuationModel(torch.nn.Module):
    def __init__(self,num_classes, sample):
        """
        :param num_classes: number of punctuation types
        :param sample: a sample batch, to find bert output size
        """
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        embedding_dim = self.get_bert_output_size(sample)
        self.before = Head(embedding_dim,num_classes)
        self.after = Head(embedding_dim,num_classes)
        self.capital = Head(embedding_dim,2)
        self.br = Head(embedding_dim,2)
        self.drop = torch.nn.Dropout(0.2)
    def get_bert_output_size(self, sample):
        # to find bert output size
        return self.bert(**sample).last_hidden_state.shape[-1]

    def forward(self, tokenized):
        x = self.bert(**tokenized).last_hidden_state
        x = self.drop(x)
        before = self.before(x)
        after = self.after(x)
        capital = self.capital(x)
        br = self.br(x)
        return before, after, capital, br
