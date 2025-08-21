import torch
import torch.nn as nn 

class Model(nn.Module):
    def __init__(self, dropout_rate: float = 0.3) -> None:
        super().__init__()

        self.feature = nn.Sequential(
            #first conv layer 
            nn.Conv2d(in_channels=1,out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #second conv layer 
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate * 0.5),
            
            #third conv layer 
            nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate * 0.5),
            
            #fourth 
            nn.Conv2d(in_channels=256,out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate * 0.5),


            nn.AdaptiveAvgPool2d(output_size=((4,4)))
        )


        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 4 * 4, out_features=2048),
            nn.BatchNorm1d(num_features=2048), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=2048, out_features=1024),
            nn.BatchNorm1d(num_features=1024), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(in_features=1024, out_features=345),
        )

        self._initialize_parameter()

    def _initialize_parameter(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

    def forward(self, input_tensor: torch.Tensor):
        output = self.feature(input_tensor)
        output = torch.flatten(output,1)
        output = self.classifier(output)

        return output
