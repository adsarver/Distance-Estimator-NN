import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class ContextHead(nn.Module):
    def __init__(self, hidden_size, lstm_num_layers, memory_length, memory_stride, img_size, out_channels=64):
        super(ContextHead, self).__init__()
        
        self.img_size = img_size
        self.out_channels = out_channels
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = hidden_size
        self.memory_length = memory_length
        self.memory_stride = memory_stride
        self.hx = None
        self.buffer = None
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self._input_dim = out_channels * 4 * 4
        
        self.lstm = nn.LSTM(
            input_size=self._input_dim,
            hidden_size=hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        
    def __get_init_hidden(self, batch_size, device, transpose=False):
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        if transpose:
            h0 = h0.transpose(0, 1).contiguous()
            c0 = c0.transpose(0, 1).contiguous()
        return (h0, c0)
    
    def __create_observation_buffer(self, batch_size, device):
        return torch.zeros(batch_size, self.memory_length, self._input_dim).to(device)
    
    def __append_to_buffer(self, new_obs):
        # new_obs: (B, feat_dim) -> unsqueeze to (B, 1, feat_dim)
        if new_obs.dim() == 2:
            new_obs = new_obs.unsqueeze(1)
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        self.buffer[:, -1:, :] = new_obs
        return self.buffer
    
    def reset_lstm(self):
        self.hx = None
        self.buffer = None
    
    def forward(self, input_image):
        if self.hx is None or self.buffer is None:
            device = input_image.device
            self.hx = self.__get_init_hidden(1, device)
            self.buffer = self.__create_observation_buffer(1, device)
            
        maps = self.cnn(input_image) # Get feature maps
        buf = self.__append_to_buffer(maps) # Update observation buffer
        lstm_out, hx_new = self.lstm(buf, self.hx) # Pass through LSTM
        self.hx = hx_new # Update hidden state
        
        return lstm_out[:, -1, :]
    
class ShapeHead(nn.Module):
    def __init__(self, hidden_size, lstm_num_layers, memory_length, memory_stride, fc_out=16):
        super(ShapeHead, self).__init__()
        
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = hidden_size
        self.memory_length = memory_length
        self.memory_stride = memory_stride
        self.hx = None
        self.buffer = None
        
        self.fc = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, fc_out),
            nn.ReLU()
        )
                
        self.lstm = nn.LSTM(
            input_size=fc_out,
            hidden_size=hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        self._input_dim = fc_out
        
    def __get_init_hidden(self, batch_size, device, transpose=False):
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        if transpose:
            h0 = h0.transpose(0, 1).contiguous()
            c0 = c0.transpose(0, 1).contiguous()
        return (h0, c0)
    
    def __create_observation_buffer(self, batch_size, device):
        return torch.zeros(batch_size, self.memory_length, self._input_dim).to(device)
    
    def __append_to_buffer(self, new_obs):
        if new_obs.dim() == 2:
            new_obs = new_obs.unsqueeze(1)
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        self.buffer[:, -1:, :] = new_obs
        return self.buffer
    
    def reset_lstm(self):
        self.hx = None
        self.buffer = None
    
    def forward(self, box):
        if self.hx is None or self.buffer is None:
            device = box.device
            self.hx = self.__get_init_hidden(1, device)
            self.buffer = self.__create_observation_buffer(1, device)
            
        proj = self.fc(box) # Get feature maps
        buf = self.__append_to_buffer(proj) # Update observation buffer
        lstm_out, hx_new = self.lstm(buf, self.hx) # Pass through LSTM
        self.hx = hx_new # Update hidden state
        
        return lstm_out[:, -1, :]
    
class ObjectHead(nn.Module):
    def __init__(self, hidden_size, lstm_num_layers, memory_length, memory_stride, out_channels=64):
        super(ObjectHead, self).__init__()
        
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = hidden_size
        self.memory_length = memory_length
        self.memory_stride = memory_stride
        self.hx = None
        self.buffer = None
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(3, 3)),  # (B, out_channels, 3, 3) — handles variable crop HxW
            nn.Flatten(),                              # (B, out_channels * 9)
        )
        
        self.lstm = nn.LSTM(
            input_size=out_channels * 9,   # 3*3 spatial grid flattened
            hidden_size=hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        self._input_dim = out_channels * 9
                
    def __get_init_hidden(self, batch_size, device, transpose=False):
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        if transpose:
            h0 = h0.transpose(0, 1).contiguous()
            c0 = c0.transpose(0, 1).contiguous()
        return (h0, c0)
    
    def __create_observation_buffer(self, batch_size, device):
        return torch.zeros(batch_size, self.memory_length, self._input_dim).to(device)
    
    def __append_to_buffer(self, new_obs):
        if new_obs.dim() == 2:
            new_obs = new_obs.unsqueeze(1)
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        self.buffer[:, -1:, :] = new_obs
        return self.buffer
    
    def reset_lstm(self):
        self.hx = None
        self.buffer = None
    
    def forward(self, crop_img):
        if self.hx is None or self.buffer is None:
            device = crop_img.device
            self.hx = self.__get_init_hidden(1, device)
            self.buffer = self.__create_observation_buffer(1, device)
            
        maps = self.cnn(crop_img) # Get feature maps
        buf = self.__append_to_buffer(maps) # Update observation buffer
        lstm_out, hx_new = self.lstm(buf, self.hx) # Pass through LSTM
        self.hx = hx_new # Update hidden state
        
        return lstm_out[:, -1, :]
    
        
class DistanceNN(nn.Module):
    def __init__(self, hidden_size, lstm_num_layers, memory_length, memory_stride, img_size, out_channels=64, fc_out=16, decode_res=8, use_obj_head=True):
        super(DistanceNN, self).__init__()
        
        self.img_size = img_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = hidden_size
        self.memory_length = memory_length
        self.memory_stride = memory_stride
        self._decode_res = decode_res
        self.use_obj_head = use_obj_head
        
        self.ctx_head   = ContextHead(hidden_size, lstm_num_layers, memory_length, memory_stride, img_size, out_channels)
        self.shape_head = ShapeHead(hidden_size, lstm_num_layers, memory_length, memory_stride, fc_out)
        self.obj_head   = ObjectHead(hidden_size, lstm_num_layers, memory_length, memory_stride, out_channels)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
        )
        
        self.pixel_decoder = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),   # one depth value per pixel
        )

        # Fixed 2-D coordinate grid for the decode resolution
        ys = torch.linspace(-1, 1, self._decode_res)
        xs = torch.linspace(-1, 1, self._decode_res)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # (decode_res*decode_res, 2)
        self.register_buffer("pos_grid", torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1))
        
    def reset_lstm(self):
        self.ctx_head.reset_lstm()
        self.shape_head.reset_lstm()
        self.obj_head.reset_lstm()

    def forward(self, img, crop_coords, crop_h, crop_w, obj_img=None, obj_dropout=0.4):
        device = img.device

        context = self.ctx_head(img)            # (1, hidden_size)
        shape   = self.shape_head(crop_coords)  # (1, hidden_size)

        if self.use_obj_head and obj_img is not None:
            obj = self.obj_head(obj_img)
            # During training, randomly zero out the object branch
            if self.training and torch.rand(1).item() < obj_dropout:
                obj = torch.zeros_like(obj)
        else:
            obj = torch.zeros(1, self.lstm_hidden_size, device=device)
        
        combined = torch.cat([context, shape, obj], dim=1)  # (1, hidden_size * 3)
        latent   = self.fc(combined)                        # (1, hidden_size)

        # Predict at a fixed small grid (DECODE_RES x DECODE_RES) then upsample
        grid = self._decode_res * self._decode_res
        latent_expanded = latent.unsqueeze(1).expand(1, grid, -1)  # (1, grid, hidden_size)
        pos = self.pos_grid.unsqueeze(0).expand(1, grid, -1)       # (1, grid, 2)
        decoder_input = torch.cat([latent_expanded, pos], dim=-1)  # (1, grid, hidden_size+2)
        depth_small = self.pixel_decoder(decoder_input).squeeze(-1)  # (1, grid)
        depth_small = depth_small.view(1, 1, self._decode_res, self._decode_res)
        
        depth_map = F.interpolate(depth_small, size=(crop_h, crop_w),
                                  mode='bilinear', align_corners=False)
        return torch.sigmoid(depth_map.squeeze(1))  # (1, crop_h, crop_w) in [0, 1]