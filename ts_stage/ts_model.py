import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
    

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], output_size)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        # print(x.shape)
        x = x.squeeze(0)
        y1 = self.tcn(x.transpose(1, 2))
        o = self.fc(y1[:, :, -1])
        return o


class YOLOv5BasedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, sequence_length):
        super(YOLOv5BasedModel, self).__init__()
        
        # Initial convolutional layer (similar to YOLOv5's initial conv)
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=0, stride=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        
        # Bottleneck layers (analogous to YOLOv5's Bottleneck)
        self.bottleneck1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # CSP Block (Cross Stage Partial Network) for deeper feature extraction
        self.csp = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # SPPF layer (SPP-Fast) to pool features across the sequence dimension
        self.sppf = nn.AdaptiveMaxPool1d(1)  # Reduces sequence_length to 1
        
        # Final fully connected layer for output
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.squeeze(0)
        # x shape: (batch, sequence_length, features)
        x = x.permute(0, 2, 1)  # Change to (batch, features, sequence_length)
        
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.bottleneck1(x) + x  # Residual connection
        x = self.csp(x) + x           # Another residual connection
        
        x = self.sppf(x).squeeze(-1)  # Pool across the sequence dimension
        
        x = self.fc(x)  # Output layer
        
        return x

