from torch import nn
from modules import ConvModule, TargetPrediction, ConvModuleSep, TargetPredictionSep, ClassPrediction

# Here we are creating four basic architectures:
# ConvNets with separate and shared weights for the two numbers and with or without an auxillary loss function
# for predicting the numbers.
# We are choosing simple architectures in order to be able to train them in reasonable time without a GPU.
# All those models consist of a convolutional feature extraction module and one or two dense prediction modules.

class ConvNetBase(nn.Module):
    #shared weights, no auxillary loss
    def __init__(self, settings):
        super(ConvNetBase, self).__init__()
        self.conv = ConvModule(settings['channels'], settings['dropout_rate'], settings['bn'])
        self.prediction = TargetPrediction(settings['channels'], settings['final_features'])

    def forward(self, x):
        x = self.conv(x)
        x = self.prediction(x)
        return x


class ConvNetSep(nn.Module):
    # separate weights, no auxillary loss
    def __init__(self, settings):
        super(ConvNetSep, self).__init__()
        self.conv = ConvModuleSep(settings['channels'], settings['dropout_rate'], settings['bn'])
        self.prediction = TargetPredictionSep(settings['channels'], settings['final_features'])

    def forward(self, x):
        x = self.conv(x)
        x = self.prediction(x)
        return x


class ConvNetBaseAux(nn.Module):
    # shared weights, auxillary loss
    def __init__(self, settings):
        super(ConvNetBaseAux, self).__init__()
        self.conv = ConvModule(settings['channels'], settings['dropout_rate'], settings['bn'])
        self.prediction = TargetPrediction(settings['channels'], settings['final_features'])
        self.class_prediction = ClassPrediction(settings['channels'], settings['hidden_features'])

    def forward(self, x):
        conv_out = self.conv(x)
        x = self.prediction(conv_out)
        classes = self.class_prediction(conv_out)
        return x, classes


class ConvNetSepAux(nn.Module):
    # separate weights, auxillary loss
    def __init__(self, settings):
        super(ConvNetSepAux, self).__init__()
        self.conv = ConvModuleSep(settings['channels'], settings['dropout_rate'], settings['bn'])
        self.prediction = TargetPredictionSep(settings['channels'], settings['final_features'])
        self.class_prediction = ClassPrediction(settings['channels'], settings['hidden_features'])

    def forward(self, x):
        conv_out = self.conv(x)
        x = self.prediction(conv_out)
        classes = self.class_prediction(conv_out)
        return x, classes