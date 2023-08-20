from sklearn.preprocessing import minmax_scale
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from time import time
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from tqdm import trange

class LSTM(nn.Module):
    def __init__(self, win_len, hidden_dim):
        super(LSTM, self).__init__()
        self.input_dim = 1
        self.win_len = win_len
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=True)
        self.hiddent2out = nn.Linear(hidden_dim, 1)

    def forward(self, seq):
        lstm_out, _ = self.lstm(seq.view(-1, self.win_len, self.input_dim))
        predict = self.hiddent2out(lstm_out)
        return predict[:, -1, :]


def train(params, dataloader, device='cpu'):
    log_interval = 1
    win_len = params.get("win_len")
    hidden_dim = params.get("z_dim")
    epoch_cnt = params.get("epoch_cnt")
    lr = params.get("lr")


    model = LSTM(win_len, hidden_dim).to(device)  #type: LSTM
    optimizer = optim.Adam(model.parameters(), lr=lr)    
    loss_fn = nn.MSELoss()
    model.train()

    loss_ls = []
    for epoch in (pbar := trange(epoch_cnt)):
        for step, x in enumerate(dataloader):
            x = x.to(device).view(-1, win_len, 1)
            x_pred = model(x)
            loss = loss_fn(x[:, -1], x_pred)
            loss_ls.append(loss.item())

            if (step + 1) % log_interval == 0:
                pbar.set_description(f"Epoch: {epoch+1}, Loss: {np.average(loss_ls): .4f}")
                loss_ls.clear()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def test(model: LSTM, dataloader, device='cpu'):
    labels, raw_seq, est_seq, loss = [], [], [], []
    model.eval()
    loss_function = nn.MSELoss()
    with torch.no_grad():
        for x, y in dataloader:
            labels.append(y.numpy())

            x = x.to(device).view(-1, model.win_len, 1)
            x_pred = model(x)

            loss.append(loss_function(x[:, -1], x_pred).cpu().item())
            raw_seq.append(x.squeeze()[:, -1].cpu().numpy())
            est_seq.append(x_pred.squeeze().cpu().numpy())

    
    raw_seq = np.concatenate(raw_seq, axis=0)
    est_seq = np.concatenate(est_seq, axis=0)
    labels = np.concatenate(labels, axis=0)

    return raw_seq, est_seq, np.average(loss), labels