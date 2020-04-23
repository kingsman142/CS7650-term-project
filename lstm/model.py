import torch
import torch.nn as nn

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class QUESTModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, lstm_layers, num_question_output, num_answer_output):
        super(QUESTModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim

        self.question_embeddings = nn.Embedding(num_embeddings = vocab_size, embedding_dim = self.embedding_dim)
        self.question_lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_size, num_layers = lstm_layers, dropout = 0.0, bidirectional = False)
        self.question_linear = nn.Linear(in_features = hidden_size, out_features = num_question_output)

        self.answer_embeddings = nn.Embedding(num_embeddings = vocab_size, embedding_dim = self.embedding_dim)
        self.answer_lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_size, num_layers = lstm_layers, dropout = 0.0, bidirectional = False)
        self.answer_linear = nn.Linear(in_features = hidden_size, out_features = num_answer_output)

    def forward(self, question, answer):
        batch_size = len(question)

        # NOTE: pytorch LSTM units take input in the form of [window_length, batch_size, num_features], which will end up being [WINDOW_SIZE, batch_size, 1] for our dataset
        # reshape question and answer sizes
        question = question.permute(1, 0)
        answer = answer.permute(1, 0)

        question_hidden_cell = (torch.zeros(1, batch_size, self.hidden_size).to(DEVICE),
                                torch.zeros(1, batch_size, self.hidden_size).to(DEVICE))
        answer_hidden_cell = (torch.zeros(1, batch_size, self.hidden_size).to(DEVICE),
                              torch.zeros(1, batch_size, self.hidden_size).to(DEVICE))

        question_embed = self.question_embeddings(question)
        question_lstm_out, _ = self.question_lstm(question_embed, question_hidden_cell)
        question_pred_scores = self.question_linear(question_lstm_out[-1])

        answer_embed = self.answer_embeddings(answer)
        answer_lstm_out, _ = self.answer_lstm(answer_embed, answer_hidden_cell)
        answer_pred_scores = self.answer_linear(answer_lstm_out[-1])

        pred_scores = torch.cat((question_pred_scores, answer_pred_scores), dim = 1)
        pred_scores = torch.sigmoid(pred_scores)

        return pred_scores
