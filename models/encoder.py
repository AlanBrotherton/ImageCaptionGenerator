import torch
import torch.nn as nn
import torchvision.models as models

# EncoderCNN class for extracting image features using a pre-trained CNN
# It uses a ResNet backbone and outputs features suitable for caption generation
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False  # freeze the backbone

        modules = list(resnet.children())[:-1]  # remove the final FC layer
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.fc(features))
        return features
