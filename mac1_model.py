import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMNetwork(nn.Module):
    def __init__(self, n_layers, hidden_dim, device):
        super(LSTMNetwork, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device

        self.lstm = nn.LSTM(input_size=11, hidden_size=self.hidden_dim, num_layers=self.n_layers)
        
#         self.lstm1 = nn.LSTM(input_size=11, hidden_size=400)
#         self.lstm2 = nn.LSTM(input_size=400, hidden_size=self.hidden_dim)
        
#         self.gru = nn.GRU(input_size=10, hidden_size=self.hidden_dim, num_layers=self.n_layers)

        self.outputlayer = nn.Linear(self.hidden_dim, 7)

    def forward(self, x, states=None):
        data_len = x.size()[0]
        data_batch = x.size()[1]

        # LSTM
        (hidden_state, cell_state) = (None, None) if states is None else states
        if hidden_state is None:
            hidden_state = torch.zeros(self.n_layers, data_batch, 400).to(self.device)
        if cell_state is None:
            cell_state = torch.zeros(self.n_layers, data_batch, 400).to(self.device)

        rnn_out, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
#         rnn1_out, (hidden_state[[0],:,:], cell_state[[0],:,:]) = self.lstm1(x, (hidden_state[[0],:,:], cell_state[[0],:,:]))
#         rnn_out, (hidden_state[[1],:,:self.hidden_dim], cell_state[[1],:,:self.hidden_dim]) = self.lstm2(rnn1_out, (hidden_state[[1],:,:self.hidden_dim], cell_state[[1],:,:self.hidden_dim]))
        states = (hidden_state, cell_state)

        # GRU
#         if states is None:
#             states = torch.zeros(self.n_layers, data_batch, self.hidden_dim).to(device)
#         rnn_out, states = self.gru(x, states)

        pred = self.outputlayer(rnn_out.view(-1, self.hidden_dim))
        return pred.view(data_len, data_batch, 7), states

    def initial_states(self):
        return None