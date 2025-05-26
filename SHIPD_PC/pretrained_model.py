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
    def __init__(self, output_channels=1):
        super(RegHeadPointNet, self).__init__()
        self.head = nn.Sequential(
            nn.Sequential(
                nn.Linear(1024, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True)
            ),
            nn.Dropout(p=0.3),
            nn.Sequential(
                nn.Linear(512, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Dropout(p=0.3),
            nn.Linear(256, output_channels)  # Directly outputs continuous values
        )

    def forward(self, x):
        return self.head(x)

class RegPointNetPT(nn.Module):
    def __init__(self, output_channels=1):
        super(RegPointNetPT, self).__init__()
        self.encoder = PointNetEncoder()
        self.prediction = RegHeadPointNet(output_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.prediction(x)
        return x