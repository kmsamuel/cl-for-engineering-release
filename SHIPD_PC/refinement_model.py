import torch
import torch.nn as nn
from pretrained_model import RegPointNetPT  # Import your original model

# class RefinementPointNet(nn.Module):
#     def __init__(self, original_pointnet):
#         super(RefinementPointNet, self).__init__()
#         # MultiStep Refinement
#         self.original_pointnet = original_pointnet
#         self.feature_size = 1024
        
#         # Feature attention
#         self.feature_attention = nn.Sequential(
#             nn.Linear(self.feature_size, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, self.feature_size),
#             nn.Sigmoid()
#         )
        
#         # First refinement stage
#         self.refine1 = nn.Sequential(
#             nn.Linear(1 + self.feature_size, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.3),
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.3),
#             nn.Linear(512, 1)
#         )
        
#         # Second refinement stage
#         self.refine2 = nn.Sequential(
#             nn.Linear(1 + self.feature_size, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.3),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 1)
#         )
    
#     def forward(self, x):
#         # Get features and initial prediction
#         features = self.original_pointnet.encoder(x)
#         initial_pred = self.original_pointnet(x)
        
#         # Apply attention
#         attention_weights = self.feature_attention(features)
#         weighted_features = features * attention_weights
        
#         # First refinement
#         combined1 = torch.cat([initial_pred, weighted_features], dim=1)
#         refinement1 = self.refine1(combined1)
#         intermediate_pred = initial_pred + refinement1
        
#         # Second refinement
#         combined2 = torch.cat([intermediate_pred, weighted_features], dim=1)
#         refinement2 = self.refine2(combined2)
#         final_pred = intermediate_pred + refinement2
        
#         return final_pred

class RefinementPointNet(nn.Module):
    def __init__(self, original_pointnet):
        super(RefinementPointNet, self).__init__()
        self.original_pointnet = original_pointnet
        
        # Feature size from the encoder
        self.feature_size = 1024
        
        # Feature attention mechanism
        self.feature_attention = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.feature_size),
            nn.Sigmoid()  # Outputs values between 0-1 for each feature
        )
        
        # Refinement layers
        self.refine = nn.Sequential(
            nn.Linear(1 + self.feature_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        # Get features from the encoder
        features = self.original_pointnet.encoder(x)
        
        # Get initial prediction
        initial_pred = self.original_pointnet(x)
        
        # Apply attention to features
        attention_weights = self.feature_attention(features)
        weighted_features = features * attention_weights
        
        # Concatenate initial prediction with weighted features
        combined = torch.cat([initial_pred, weighted_features], dim=1)
        
        # Generate refinement
        refinement = self.refine(combined)
        
        # Final prediction with skip connection
        final_pred = initial_pred + refinement
        
        return final_pred