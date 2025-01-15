import torch.nn as nn
import torch

class CNNLSTM_Model(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers):
        super(CNNLSTM_Model, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers

        self.c1 = nn.Conv1d(
            in_channels=n_features,
            out_channels= 19,
            kernel_size = 2,
            stride = 1
        )

        self.c2 = nn.Conv1d(
            in_channels=19,
            out_channels= 38,
            kernel_size = 2,
            stride = 1
        )

        self.c3 = nn.Conv1d(
            in_channels=38,
            out_channels= 76,
            kernel_size = 2,
            stride = 1
        )

        self.bn = nn.BatchNorm1d(20) 

        self.lstm = nn.LSTM(
            input_size= 76,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)

    def reset_hidden_state(self, batch_size):
        self.hidden = (
            torch.zeros(self.n_layers, batch_size, self.n_hidden),
            torch.zeros(self.n_layers, batch_size, self.n_hidden)
        )

    def forward(self, sequences):
        sequences = sequences.permute(0, 2, 1)
        conv_out = self.c1(sequences)
        conv_out = torch.relu(conv_out)  # 활성화 함수
        conv_out = self.c2(conv_out)  # 두 번째 Conv1D
        conv_out = torch.relu(conv_out)  # 활성화 함수
        conv_out = self.c3(conv_out)  # 두 번째 Conv1D
        conv_out = torch.relu(conv_out)  # 활성화 함수

        conv_out = conv_out.permute(0, 2, 1)  # Conv1D 결과에서 channel dimension 제거

        # LSTM에 데이터 전달
        lstm_out, _ = self.lstm(conv_out)
        lstm_out = self.dropout(lstm_out)
        last_time_step = lstm_out[:, -1, :]  # 마지막 time step 추출
        
        y_pred = self.linear(last_time_step)
        return y_pred