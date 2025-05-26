import torch
import torch.nn as nn

class Regression_ResNet_Model(torch.nn.Module):
    def __init__(self, Reg_Dict):
        nn.Module.__init__(self)
        
        self.xdim = Reg_Dict['xdim']
        self.ydim = Reg_Dict['ydim']
        self.tdim = Reg_Dict['tdim']
        self.net = Reg_Dict['net']
        
        self.fc = nn.ModuleList()
        
        self.fc.append(self.LinLayer(self.tdim,self.net[0]))
        
        for i in range(1, len(self.net)):
            self.fc.append(self.LinLayer(self.net[i-1],self.net[i]))
            
        self.fc.append(self.LinLayer(self.net[-1], self.tdim))
        
        
        self.finalLayer = nn.Sequential(nn.Linear(self.tdim, self.ydim))
        
    
        self.X_embed = nn.Linear(self.xdim, self.tdim)
       
        
    def LinLayer(self, dimi, dimo):
        
        return nn.Sequential(nn.Linear(dimi,dimo),
                             nn.SiLU(),
                             nn.LayerNorm(dimo),
                             nn.Dropout(p=0.2))
        


    def forward(self, x):
        x = self.X_embed(x) 
        res_x = x
        
        for i in range(0,len(self.fc)):
            x = self.fc[i](x)
        
        x = torch.add(x,res_x)
        
        x = self.finalLayer(x)
        
        return x
