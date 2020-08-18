import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMNetwork(nn.Module):
    def __init__(self, n_layers, hidden_dim, device):
        super(LSTMNetwork, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim1 = hidden_dim[0]
        self.hidden_dim2 = hidden_dim[1]
        self.hidden_dim_a = 192
        self.device = device
        self.train_initial_memory = False

#         self.lstm = nn.LSTM(input_size=11, hidden_size=self.hidden_dim, num_layers=self.n_layers)
        
        self.lstm1 = nn.LSTM(input_size=9, hidden_size=self.hidden_dim1)
        self.lstm2 = nn.LSTM(input_size=self.hidden_dim1, hidden_size=self.hidden_dim2)
        self.lstm_a1 = nn.LSTM(input_size=9, hidden_size=self.hidden_dim_a)
        self.lstm_a2 = nn.LSTM(input_size=self.hidden_dim_a, hidden_size=self.hidden_dim_a)
        
#         self.gru = nn.GRU(input_size=10, hidden_size=self.hidden_dim, num_layers=self.n_layers)

        self.outputlayer = nn.Linear(self.hidden_dim2, 5)
        self.outputlayer_a = nn.Linear(self.hidden_dim_a, 2)
        
        self.init_hidden_state_1 = torch.zeros((1, 1, self.hidden_dim1), dtype=torch.float32, requires_grad=self.train_initial_memory, device=self.device)
        self.init_cell_state_1 = torch.zeros((1, 1, self.hidden_dim1), dtype=torch.float32, requires_grad=self.train_initial_memory, device=self.device)
        self.init_hidden_state_2 = torch.zeros((1, 1, self.hidden_dim2), dtype=torch.float32, requires_grad=self.train_initial_memory, device=self.device)
        self.init_cell_state_2 = torch.zeros((1, 1, self.hidden_dim2), dtype=torch.float32, requires_grad=self.train_initial_memory, device=self.device)
        
        self.init_hidden_state_a1 = torch.zeros((1, 1, self.hidden_dim_a), dtype=torch.float32, requires_grad=self.train_initial_memory, device=self.device)
        self.init_cell_state_a1 = torch.zeros((1, 1, self.hidden_dim_a), dtype=torch.float32, requires_grad=self.train_initial_memory, device=self.device)
        self.init_hidden_state_a2 = torch.zeros((1, 1, self.hidden_dim_a), dtype=torch.float32, requires_grad=self.train_initial_memory, device=self.device)
        self.init_cell_state_a2 = torch.zeros((1, 1, self.hidden_dim_a), dtype=torch.float32, requires_grad=self.train_initial_memory, device=self.device)
        if self.train_initial_memory:
            self.init_hidden_state_1 = nn.Parameter(self.init_hidden_state_1)
            self.init_cell_state_1 = nn.Parameter(self.init_cell_state_1)
            self.init_hidden_state_2 = nn.Parameter(self.init_hidden_state_2)
            self.init_cell_state_2 = nn.Parameter(self.init_cell_state_2)
            self.init_hidden_state_a1 = nn.Parameter(self.init_hidden_state_a1)
            self.init_cell_state_a1 = nn.Parameter(self.init_cell_state_a1)
            self.init_hidden_state_a2 = nn.Parameter(self.init_hidden_state_a2)
            self.init_cell_state_a2 = nn.Parameter(self.init_cell_state_a2)
            
        
    def forward(self, x, states=None):
        data_len = x.size()[0]
        data_batch = x.size()[1]

        # LSTM
        if states is None:
#             hidden_state, cell_state = torch.zeros((self.n_layers, data_batch, 400), dtype=torch.float32, device=self.device), torch.zeros((self.n_layers, data_batch, 400), dtype=torch.float32, device=self.device)
            hidden_state_1 = self.init_hidden_state_1.repeat(1,data_batch,1)
            cell_state_1 = self.init_cell_state_1.repeat(1,data_batch,1)
            hidden_state_2 = self.init_hidden_state_2.repeat(1,data_batch,1)
            cell_state_2 = self.init_cell_state_2.repeat(1,data_batch,1)
            hidden_state_a1 = self.init_hidden_state_a1.repeat(1,data_batch,1)
            cell_state_a1 = self.init_cell_state_a1.repeat(1,data_batch,1)
            hidden_state_a2 = self.init_hidden_state_a2.repeat(1,data_batch,1)
            cell_state_a2 = self.init_cell_state_a2.repeat(1,data_batch,1)
        else:
            (hidden_state_1, cell_state_1, hidden_state_2, cell_state_2, hidden_state_a1, cell_state_a1, hidden_state_a2, cell_state_a2) = (states['h1'], states['c1'], states['h2'], states['c2'], states['ha1'], states['ca1'], states['ha2'], states['ca2'])
#             (hidden_state, cell_state) = (None, None) if states is None else states

#         rnn_out, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
        rnn1_out, (hidden_state_1, cell_state_1) = self.lstm1(x, (hidden_state_1, cell_state_1))
        rnn2_out, (hidden_state_2, cell_state_2) = self.lstm2(rnn1_out, (hidden_state_2, cell_state_2))
        rnn_a1_out, (hidden_state_a1, cell_state_a1) = self.lstm_a1(x, (hidden_state_a1, cell_state_a1))
        rnn_a2_out, (hidden_state_a2, cell_state_a2) = self.lstm_a2(rnn_a1_out, (hidden_state_a2, cell_state_a2))

#         states = (hidden_state, cell_state)
        states = {'h1': hidden_state_1, 'c1': cell_state_1, 'h2': hidden_state_2, 'c2': cell_state_2, 'ha1': hidden_state_a1, 'ca1': cell_state_a1, 'ha2': hidden_state_a2, 'ca2': cell_state_a2}

        # GRU
#         if states is None:
#             states = torch.zeros(self.n_layers, data_batch, self.hidden_dim).to(device)
#         rnn_out, states = self.gru(x, states)

        pred = self.outputlayer(rnn2_out.view(-1, self.hidden_dim2)).view(data_len, data_batch, 5)
        pred_a = self.outputlayer_a(rnn_a2_out.view(-1, self.hidden_dim_a)).view(data_len, data_batch, 2)

        out = torch.zeros(data_len, data_batch, 7).to(self.device)
        out[:,:,[0,3,4,5,6]] = pred
        out[:,:,[1,2]] = pred_a
        return out, states

    def initial_states(self):
        return None