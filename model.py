import torch
import torch.functional as F
from torch import nn

class DistanceNN(nn.Module):
    def __init__(self, hidden_size, img_size, depth_stride):
        super(DistanceNN, self).__init__()
        
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
        
        lstm_input = self.__get_lstm_input_size()
        
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        # Fully connected layers, takes LSTM output and gives distance value for every depth_stride pixels
        self.nn = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_size * img_size)
        )
        
    def __get_lstm_input_size(self):
        dummy_input = torch.zeros(1, 3, 64, 64)  # Assuming input images are 64x64 RGB
        cnn_output = self.cnn(dummy_input)
        return cnn_output.shape[1]

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x