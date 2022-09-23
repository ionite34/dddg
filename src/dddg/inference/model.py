import torch
import torch.nn as nn
import torch.nn.functional as nf
import torchvision.models as models


class MultiOutputModel(nn.Module):
    def __init__(self, n_hat_classes, n_acc_classes):
        super().__init__()
        self.base_model = models.mobilenet_v2().features  # model without classifier
        last_channel = models.mobilenet_v2().last_channel  # size of the layer before classifier

        # spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for outputs
        self.hat = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_hat_classes)
        )
        self.acc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_acc_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        return {
            'hat': self.hat(x),
            'acc': self.acc(x),
        }

    @staticmethod
    def get_loss(net_output, ground_truth):
        hat_loss = nf.cross_entropy(net_output['hat'], ground_truth['hat_labels'])
        acc_loss = nf.cross_entropy(net_output['acc'], ground_truth['acc_labels'])

        loss = hat_loss + acc_loss
        return loss, {'hat': hat_loss, 'acc': acc_loss}
