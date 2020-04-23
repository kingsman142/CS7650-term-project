import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import QUESTDataset
from model import QUESTModel

BATCH_SIZE = 1
LEARNING_RATE = 0.01
NUM_EPOCHS = 15
HIDDEN_SIZE = 100 # 50 works great with 1 sample
LSTM_LAYERS = 1
EMBEDDING_DIM = 100 # 100 works great with 1 sample
WORD2IDX_PATH = os.path.join("word2idx.json")

QUESTION_OUTPUT_SIZE = 21 # 21 features/attributes corresponding to the questions
ANSWER_OUTPUT_SIZE = 9 # 9 features/attributes corresponding to the answers

MODEL_SAVE_NAME = "epoch_{}_batchsize_{}_lr_{}_hiddensize_{}_lstmlayers_{}_loss_{}"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Training on {}...".format(DEVICE))

# load in word2idx and determine vocab size
with open(WORD2IDX_PATH, "r") as word2idx_file:
    word2idx = json.loads(word2idx_file.read())
    vocab_size = len(word2idx)

# init dataset and loaders for train/val/test splits
train_data = QUESTDataset(data_path = os.path.join("train_preprocessed.csv"))
val_data = QUESTDataset(data_path = os.path.join("val_preprocessed.csv"))
test_data = QUESTDataset(data_path = os.path.join("test_preprocessed.csv"))
train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = 1)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1)

# init training modules, such as the model, Adam optimizer and loss function
model = QUESTModel(vocab_size = vocab_size, embedding_dim = EMBEDDING_DIM, hidden_size = HIDDEN_SIZE, lstm_layers = LSTM_LAYERS, num_question_output = QUESTION_OUTPUT_SIZE, num_answer_output = ANSWER_OUTPUT_SIZE).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE)
loss_function = nn.MSELoss(reduction = 'sum').to(DEVICE)

# create a 'models' directory to save models
if not os.path.exists("models"):
    os.mkdir("models")

# print out debugging statistics
print("Training model with:\n\tEpochs: {}\n\tLoss function: {}\n\tLearning rate: {}\n\tBatch size: {}\n\tHidden size: {}\n\tVocab size: {}".format(NUM_EPOCHS, loss_function, LEARNING_RATE, BATCH_SIZE, HIDDEN_SIZE, vocab_size))

best_model_loss = None
best_model_epoch = None
best_model_path = None
for epoch in range(NUM_EPOCHS):
    # train model for one epoch
    train_loss = 0.0
    for batch_id, samples in enumerate(train_loader):
        # make predictions for the samples in this batch
        question, answer, true_scores = samples["question"].to(DEVICE), samples["answer"].to(DEVICE), samples["scores"].to(DEVICE)
        pred_scores = model(question, answer)

        # calculate loss
        loss = loss_function(pred_scores, true_scores)
        train_loss += loss.item()

        # backpropagation
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 500 == 0:
            print("TRAIN -- EPOCH {}/{} -- BATCH {}/{} -- Avg. Loss: {}".format(epoch+1, NUM_EPOCHS, batch_id+1, len(train_loader), loss.item() / BATCH_SIZE))
    print("Average train loss: {}".format(train_loss / (len(train_loader) * BATCH_SIZE)))

    # test model on validation set
    avg_val_loss = 0.0
    for batch_id, samples in enumerate(val_loader):
        question, answer, true_scores = samples["question"].to(DEVICE), samples["answer"].to(DEVICE), samples["scores"].to(DEVICE)
        pred_scores = model(question.to(DEVICE), answer.to(DEVICE))

        loss = loss_function(pred_scores, true_scores)
        avg_val_loss += loss.item()

        if batch_id % 50 == 0:
            print("VALIDATION -- EPOCH {}/{} -- BATCH {}/{} -- Avg. Loss: {}".format(epoch+1, NUM_EPOCHS, batch_id+1, len(val_loader), loss.item()))
    avg_val_loss /= len(val_loader)
    print("VALIDATION -- Avg. Loss: {}".format(avg_val_loss))
    if best_model_loss is None or best_model_loss > avg_val_loss: # if this val loss is loss than the previous best model, save it to disk
        # remove previous best model
        if best_model_path is not None and os.path.exists(best_model_path):
            os.remove(best_model_path)
        print("Found new best model! Previous best at epoch {} with loss {}".format(best_model_epoch, best_model_loss))

        # save new best model
        best_model_loss = avg_val_loss
        best_model_epoch = epoch+1
        best_model_path = os.path.join("models", MODEL_SAVE_NAME.format(best_model_epoch, BATCH_SIZE, LEARNING_RATE, HIDDEN_SIZE, LSTM_LAYERS, best_model_loss))
        torch.save(model.state_dict(), best_model_path)
print("Best model found at epoch {}...".format(best_model_epoch))
