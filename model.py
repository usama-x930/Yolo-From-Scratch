import torch
import torch.nn as nn

from cfg import config
from utils import intersection_over_union


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.bn(self.conv(x)))
    
 
class My_Model(nn.Module):
    def __init__(self, in_channels = 3, **kwargs):
        super(My_Model, self).__init__()
        self.architecture = config()
        self.in_channels = in_channels

        self.darknet = self.create_darknet(self.architecture)
        self.fc = self.create_fc(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fc(torch.flatten(x, start_dim=1))
    
    def create_darknet(self, architecture):
        layers = []
        in_channels = self.in_channels

        for layer in architecture:
            # if isinstance(layer, tuple):
            if type(layer) == tuple:
                # layers.append(CNNBlock(in_channels, layer[1], kernel_size=layer[0], stride=layer[2], padding=layer[3]))
                layers += [
                    CNNBlock(in_channels, layer[1], kernel_size=layer[0], stride=layer[2], padding=layer[3])
                ]
                in_channels = layer[1]

            elif type(layer) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(layer) == list:
                for i in range(layer[-1]):
                    layers += [
                        CNNBlock(in_channels, layer[0][1], kernel_size = layer[0][0], stride=layer[0][2], padding=layer[0][3])
                    ]
                    layers += [
                        CNNBlock(layer[0][1], layer[1][1], kernel_size = layer[1][0], stride=layer[1][2], padding=layer[1][3])
                    ]

                    in_channels = layer[1][1]    
                    
        return nn.Sequential(*layers)

            
    def create_fc(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5))
        )



class LossFunction(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(LossFunction, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.mse = nn.MSELoss(reduction='sum')
        self.lambda_noobj = 0.5
        self.lambda_coord = 5.0

    def forward(self, predictions, targets):
        predictions = predictions.view(-1, self.S, self.S, self.B * 5 + self.C)

        iou_b1 = intersection_over_union(predictions[..., 21:25], targets[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], targets[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_max, best_box = torch.max(ious, dim=0)
        exist_box = targets[..., 20].unsqueeze(3)

        # FOR BOX COORDINATES
        box_predictions = exist_box * ((best_box * predictions[..., 26:30] + (1 - best_box) * predictions[..., 21:25]) )
        box_targets = exist_box * targets[..., 21:25]
        box_predictions_xy = box_predictions[..., 2:4]
        box_targets_xy = box_targets[..., 2:4]

        box_predictions = box_predictions.clone()
        box_targets = box_targets.clone()

        box_predictions[..., 2:4] = torch.sign(box_predictions_xy) * torch.sqrt(torch.abs(box_predictions_xy) + 1e-6)
        box_targets[..., 2:4] = torch.sqrt(torch.abs(box_targets_xy) + 1e-6)

        box_mse = self.mse(torch.flatten(box_predictions, end_dim=-2), torch.flatten(box_targets, end_dim=-2))



        # FOR OBJECT LOSS
        pred_box = (best_box * predictions[..., 25:26] + (1-best_box) * predictions[..., 20:21])
        object_loss = self.mse(torch.flatten(exist_box * pred_box), torch.flatten(exist_box * targets[..., 20:21]))
        # FOR NO OBJECT LOSS
        no_object_loss = self.mse(torch.flatten((1 - exist_box) * predictions[..., 20:21], start_dim=1), torch.flatten((1 - exist_box) * targets[..., 20:21], start_dim=1))
        no_object_loss += self.mse(torch.flatten((1 - exist_box) * predictions[..., 25:26], start_dim=1), torch.flatten((1 - exist_box) * targets[..., 20:21], start_dim=1))
        # FOR CLASS LOSS
        class_loss = self.mse(
            torch.flatten(exist_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exist_box * targets[..., :20], end_dim=-2,),
        )
        loss = (
            self.lambda_coord * box_mse
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        return loss

