import torch
import torch.nn as nn

class stacked2d(nn.Module):
    def __init__(
        self, 
        in_channels: int = 3,
        hidden_channels: int = 64, 
        out_channels: int = 3,
        kernel_size: int = 3
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
        )

        self.conv3 = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
        )

        # Define max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define ReLU activation function
        self.relu = nn.ReLU(inplace=True)

        # Calculate the number of features after pooling
        self.num_features = self.hidden_channels * 14 * 14  

        self.fc1 = nn.Linear(
            in_features=self.num_features, 
            out_features=self.hidden_channels
        )

        self.fc2 = nn.Linear(
            in_features=self.hidden_channels, 
            out_features=self.out_channels
        )


    def forward(self, x):
        # Apply convolutional layers with ReLU activation and max-pooling
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        # Flatten the output
        x = x.view(-1, self.num_features)
        # Apply fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x  