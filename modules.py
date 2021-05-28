import torch
from torch import nn

class ConvModule(nn.Module):
    #feature extraction with shared weights
    def __init__(self, channels=16, dropout_rate=0, bn=False):
        super(ConvModule, self).__init__()
        in_out = [(1, channels // 2), (channels // 2, channels), (channels, channels)]
        self.conv = nn.ModuleList(
            [nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3)) for in_c, out_c in in_out])
        self.bn = nn.ModuleList([nn.BatchNorm2d(out_c) for in_c, out_c in in_out])

        self.dropout = nn.Dropout(dropout_rate)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.relu = nn.ReLU()

        self.use_bn = bn

    def forward(self, x):
        N = x.shape[0]
        x = x.reshape(-1, 1, 14, 14)
        for conv, bn in zip(self.conv, self.bn):
            x = conv(x)
            x = self.relu(x)
            if self.use_bn:
                x = bn(x)
            x = self.maxpool(x)
        x = self.dropout(x)
        x = x.reshape(2 * N, -1)
        return x


class ConvModuleSep(nn.Module):
    # feature extraction with separate weights
    def __init__(self, channels=16, dropout_rate=0, bn=False):
        super(ConvModuleSep, self).__init__()
        in_out = [(2, channels), (channels, channels * 2), (channels * 2, channels * 2)]
        self.conv = nn.ModuleList(
            [nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), groups=2) for in_c, out_c in in_out])
        self.bn = nn.ModuleList([nn.BatchNorm2d(out_c) for in_c, out_c in in_out])

        self.dropout = nn.Dropout(dropout_rate)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.relu = nn.ReLU()

        self.use_bn = bn

    def forward(self, x):
        N = x.shape[0] // 2
        for conv, bn in zip(self.conv, self.bn):
            x = conv(x)
            x = self.relu(x)
            if self.use_bn:
                x = bn(x)
            x = self.maxpool(x)
        x = self.dropout(x)
        x = x.reshape(2 * N, -1)
        return x


class TargetPrediction(nn.Module):
    # dense target prediction with shared weights
    def __init__(self, channels, final_features):
        super(TargetPrediction, self).__init__()
        self.fc1 = nn.Linear(in_features=channels, out_features=final_features)
        self.fc2 = nn.Linear(in_features=final_features * 2, out_features=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N = x.shape[0] // 2
        x = x.reshape(2 * N, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = x.reshape(N, -1)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class TargetPredictionSep(nn.Module):
    # dense target prediction with separate weights
    def __init__(self, channels, final_features):
        super(TargetPredictionSep, self).__init__()
        self.fc1_1 = nn.Linear(in_features=channels, out_features=final_features)
        self.fc1_2 = nn.Linear(in_features=channels, out_features=final_features)
        self.fc2 = nn.Linear(in_features=final_features * 2, out_features=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N = x.shape[0] // 2
        x = x.reshape(N, 2, -1)
        x_1 = self.fc1_1(x[:, 0, :])
        x_2 = self.fc1_2(x[:, 1, :])
        x = torch.stack([x_1, x_2], dim=1).reshape(N, -1)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class ClassPrediction(nn.Module):
    # class prediction for auxillary loss
    def __init__(self, channels=16, hidden_features=16):
        super(ClassPrediction, self).__init__()
        self.fc_class_1 = nn.Linear(in_features=channels, out_features=hidden_features)
        self.fc_class_2 = nn.Linear(in_features=hidden_features, out_features=10)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        N = x.shape[0] // 2
        classes = x.reshape(2 * N, -1)
        classes = self.fc_class_1(classes)
        classes = self.relu(classes)
        classes = self.fc_class_2(classes)
        classes = self.softmax(classes)
        return classes