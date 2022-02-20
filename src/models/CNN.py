import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNModel, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv3d(7, 32, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2))
        )

        self.linear_block = nn.Sequential(
            nn.Linear(96000000, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.15),
            nn.Linear(128, num_classes)
        )        

    def forward(self, x):
        out = self.conv_block(x)
        out = out.view(out.size(0), -1) 
        return self.linear_block(out)