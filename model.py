import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class SpatialDecoder(nn.Module):
    """Decode depth from CNN spatial features + LSTM latent vector.

    Takes spatial features from ObjectHead's CNN (64ch @ 16×16) and the
    combined LSTM latent (128-d, tiled spatially).  Two upsample steps
    bring features from 16×16 → 64×64, then a final bilinear resize to
    the requested crop resolution.
    """

    def __init__(self, latent_dim, feat_ch=64, **kwargs):
        super().__init__()
        in_ch = feat_ch + latent_dim  # 64 + 128 = 192

        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 2, 3, 1, 1),  # 2 = depth + log_uncertainty
        )

    def forward(self, spatial_feat, latent, out_h, out_w):
        """
        spatial_feat: (B, feat_ch, 16, 16)  — from ObjectHead CNN
        latent:       (B, latent_dim)        — combined LSTM hidden state
        Returns:      (B, 2, out_h, out_w)   — channel 0 = depth, channel 1 = log_unc
        """
        B, _, H, W = spatial_feat.shape
        lat = latent.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
        x = torch.cat([spatial_feat, lat], dim=1)   # (B, feat_ch+latent, 16, 16)
        x = self.decoder(x)                          # (B, 2, 64, 64)
        if x.shape[2] != out_h or x.shape[3] != out_w:
            x = F.interpolate(x, (out_h, out_w), mode='bilinear', align_corners=False)
        return x

class ContextHead(nn.Module):
    def __init__(self, hidden_size, lstm_num_layers, img_size, out_channels=64, avg_pool_size=(8, 8)):
        super(ContextHead, self).__init__()
        
        self.img_size = img_size
        self.out_channels = out_channels
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = hidden_size
        self.hx = None
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(avg_pool_size),
            nn.Flatten(),
        )
        self._input_dim = out_channels * avg_pool_size[0] * avg_pool_size[1]
        
        self.lstm = nn.LSTM(
            input_size=self._input_dim,
            hidden_size=hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        
    def _get_init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)
    
    def reset_lstm(self):
        self.hx = None
    
    def forward(self, input_image):
        if self.hx is None:
            self.hx = self._get_init_hidden(input_image.shape[0], input_image.device)
            
        maps = self.cnn(input_image)            # (B, feat_dim)
        maps = maps.unsqueeze(1)                 # (B, 1, feat_dim) — single timestep
        lstm_out, self.hx = self.lstm(maps, self.hx)
        
        return lstm_out[:, -1, :]
    
class ShapeHead(nn.Module):
    def __init__(self, hidden_size, lstm_num_layers, fc_out=16):
        super(ShapeHead, self).__init__()
        
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = hidden_size
        self.hx = None
        
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
        
    def _get_init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)
    
    def reset_lstm(self):
        self.hx = None
    
    def forward(self, box):
        if self.hx is None:
            self.hx = self._get_init_hidden(box.shape[0], box.device)
            
        proj = self.fc(box)                      # (B, fc_out)
        proj = proj.unsqueeze(1)                  # (B, 1, fc_out) — single timestep
        lstm_out, self.hx = self.lstm(proj, self.hx)
        
        return lstm_out[:, -1, :]
    
class ObjectHead(nn.Module):
    FEAT_POOL_SIZE = 16  # spatial resolution of features passed to decoder

    def __init__(self, hidden_size, lstm_num_layers, out_channels=64, avg_pool_size=(8, 8)):
        super(ObjectHead, self).__init__()
        
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = hidden_size
        self.hx = None
        
        # Conv layers that produce spatial feature maps
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Pool path for LSTM (unchanged from before)
        self.pool = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=avg_pool_size),
            nn.Flatten(),
        )

        # Fixed-size pool for spatial features sent to the decoder
        self.feat_pool = nn.AdaptiveAvgPool2d(self.FEAT_POOL_SIZE)

        self._input_dim = out_channels * avg_pool_size[0] * avg_pool_size[1]
        
        self.lstm = nn.LSTM(
            input_size=self._input_dim,
            hidden_size=hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
                
    def _get_init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)
    
    def reset_lstm(self):
        self.hx = None
    
    def forward(self, crop_img):
        if self.hx is None:
            self.hx = self._get_init_hidden(crop_img.shape[0], crop_img.device)

        feat_maps = self.features(crop_img)      # (B, 64, H/4, W/4)

        # LSTM path (unchanged)
        pooled = self.pool(feat_maps)            # (B, 64*8*8)
        pooled = pooled.unsqueeze(1)             # (B, 1, feat_dim)
        lstm_out, self.hx = self.lstm(pooled, self.hx)

        # Spatial features for decoder
        spatial_feat = self.feat_pool(feat_maps) # (B, 64, 16, 16)
        
        return lstm_out[:, -1, :], spatial_feat
    
class DistanceNN(nn.Module):
    def __init__(self, hidden_size, lstm_num_layers, img_size, out_channels=96, fc_out=16, use_obj_head=True, **kwargs):
        super(DistanceNN, self).__init__()
        
        self.img_size = img_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = hidden_size
        self.use_obj_head = use_obj_head
        
        self.ctx_head   = ContextHead(hidden_size, lstm_num_layers, img_size, out_channels)
        self.shape_head = ShapeHead(hidden_size, lstm_num_layers, fc_out)
        self.obj_head   = ObjectHead(hidden_size, lstm_num_layers, out_channels)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LeakyReLU(0.2),
        )
        
        self.obj_feat_ch = out_channels  # channels from ObjectHead CNN
        self.decoder = SpatialDecoder(hidden_size, feat_ch=out_channels)

        # Predicts absolute depth scale (mm) from latent — used to recover
        # real-world distances from the [0,1] normalised depth map.
        self.scale_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus(),          # ensures positive output
        )
        
    def reset_lstm(self):
        self.ctx_head.reset_lstm()
        self.shape_head.reset_lstm()
        self.obj_head.reset_lstm()

    def forward(self, img, crop_coords, crop_h, crop_w, obj_img=None, obj_dropout=0.4):
        B = img.shape[0]
        device = img.device

        context = self.ctx_head(img)            # (B, hidden_size)
        shape   = self.shape_head(crop_coords)  # (B, hidden_size)

        if self.use_obj_head and obj_img is not None:
            obj, spatial_feat = self.obj_head(obj_img)  # (B, hidden_size), (B, 64, 16, 16)
        else:
            obj = torch.zeros(B, self.lstm_hidden_size, device=device)
            spatial_feat = torch.zeros(B, self.obj_feat_ch,
                                       ObjectHead.FEAT_POOL_SIZE,
                                       ObjectHead.FEAT_POOL_SIZE, device=device)
        
        combined = torch.cat([context, shape, obj], dim=1)  # (1, hidden_size * 3)
        latent = self.fc(combined)                           # (1, hidden_size)

        depth_map = self.decoder(spatial_feat, latent, crop_h, crop_w)  # (1, 2, crop_h, crop_w)

        depth   = torch.sigmoid(depth_map[:, 0:1])           # (1, 1, H, W) in [0, 1]
        log_unc = depth_map[:, 1:2].clamp(-10, 10)           # (1, 1, H, W) clamped for stability

        pred_scale = self.scale_head(latent).squeeze(-1)      # (B,) predicted depth scale in mm

        return depth.squeeze(1), log_unc.squeeze(1), pred_scale  # (1,H,W), (1,H,W), (B,)