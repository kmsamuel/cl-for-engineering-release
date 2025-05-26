import torch
import torch.nn as nn
import torch.nn.functional as F

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=-1, keepdim=False)[0]
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x

class STNkd(nn.Module):
    def __init__(self):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(64, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4096)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=-1, keepdim=False)[0]
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d()
        self.conv0_1 = nn.Conv1d(4, 64, 1)
        self.conv0_2 = nn.Conv1d(64, 64, 1)
        self.conv1 = nn.Conv1d(64, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn0_1 = nn.BatchNorm1d(64)
        self.bn0_2 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fstn = STNkd()

    def forward(self, x):
        x = self.bn0_1(self.conv0_1(x))
        x = self.bn0_2(self.conv0_2(x))
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        # print('x size right before error: ', x.shape)
        # print(f"Memory used: {x.element_size() * x.nelement() / 1024**3:.2f} GB")
        x = self.bn3(self.conv3(x))
        x = x.max(dim=-1, keepdim=False)[0]
        return x

class RegHeadPointNet(nn.Module):
    def __init__(self, output_channels=1, condition_dim=3): # when predicting just one target value
    # def __init__(self, output_channels=3, condition_dim=3): # when predicting all target values

        super(RegHeadPointNet, self).__init__()
        
        # Keep original head exactly as it was for pretrained weights
        self.head = nn.Sequential(
            nn.Sequential(
                nn.Linear(1024, 512, bias=False),  # Keep original dimensions
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True)
            ),
            nn.Dropout(p=0.3),
            nn.Sequential(
                nn.Linear(512, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True),  # Try LeakyReLU instead of ReLU
            ),
            nn.Dropout(p=0.3),
            nn.Linear(256, 256)  # Modified to output features instead of final prediction
        )
        
        # Process flight conditions separately
        self.fc_conditions = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True)
        )
        
        # New final layer to combine both pathways
        self.final_layer = nn.Linear(256 + 128, output_channels)

    def forward(self, x, flt_conds):
        # Process point features through original head
        x = self.head(x)
        
        # Process flight conditions
        condition_features = self.fc_conditions(flt_conds)
        
        # Combine features and make final prediction
        combined = torch.cat([x, condition_features], dim=1)
        return self.final_layer(combined)

class RegPointNetPT(nn.Module):
    def __init__(self, output_channels=1, condition_dim=3):
        super(RegPointNetPT, self).__init__()
        self.encoder = PointNetEncoder()
        self.prediction = RegHeadPointNet(output_channels, condition_dim)

    def forward(self, inputs):
        # Handle different input formats
        if isinstance(inputs, dict):
            # If inputs is a dictionary from the modified dataset
            point_cloud = inputs['point_cloud']
            flt_conds = inputs['flight_conditions']

        elif isinstance(inputs, tuple) and len(inputs) == 2:
            # If inputs is a tuple of (point_cloud, flt_conds)
            point_cloud, flt_conds = inputs
        
        else:
            # Legacy format or incorrect input
            raise ValueError("Input format not recognized. Expected dict or tuple.")
        
        # Process through encoder and prediction head
        features = self.encoder(point_cloud)

        output = self.prediction(features, flt_conds)
        
        return output