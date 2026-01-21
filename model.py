import torch
import torch.functional as F
from torch import nn
import numpy as np

class DistanceNN(nn.Module):
    def __init__(self, hidden_size, lstm_num_layers, memory_length, batch_size, memory_stride, img_size):
        super(DistanceNN, self).__init__()
        
        self.img_size = img_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = hidden_size
        self.memory_length = memory_length
        self.memory_stride = memory_stride
        self.batch_size = batch_size
        self.hx = self.__get_init_hidden(batch_size, 'cuda' if torch.cuda.is_available() else 'cpu')
        self.buffer = self.__create_observation_buffer(batch_size, 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        lstm_input = self.__get_lstm_input_size(img_size)
        
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
                
        # Fully connected layers, takes LSTM output and gives distance value for every depth_stride pixels
        self.nn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
    def __get_init_hidden(self, batch_size, device, transpose=False):
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        if transpose:
            h0 = h0.transpose(0, 1).contiguous()
            c0 = c0.transpose(0, 1).contiguous()
        return (h0, c0)
    
    def __create_observation_buffer(self, batch_size, device):
        return torch.zeros(batch_size, self.memory_length, self.lstm_hidden_size).to(device)
    
    def __append_to_buffer(self, new_obs):        
        # Roll the buffer to make room for new observations
        self.buffer = torch.roll(self.buffer, shifts=-self.memory_stride, dims=1)
        
        # Add new observations at the end
        self.buffer[:, -self.memory_stride:, :] = new_obs[:, :self.memory_stride, :]
        
        return self.buffer
        
        

    def forward(self, x, crop_coords=None):
        x = self.cnn(x) # Get feature maps
        combined_features = torch.cat([x, crop_coords], dim=1)  # Combine all features
        buf = self.__append_to_buffer(combined_features.unsqueeze(1)) # Update observation buffer
        x, hx = self.lstm(buf, self.hx) # Pass through LSTM
        self.hx = hx # Update hidden state
        x = self.nn(x[:, -1, :]) # Pass through fully connected layers
        
        return x