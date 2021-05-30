import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


class Image_VGG_16(torch.nn.Module):

    def __init__(self, require_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=False).features
        self.slice = torch.nn.Sequential()
        # conv_5_1
        for x in range(26):
            self.slice.add_module(str(x), vgg_pretrained_features[x])

        if not require_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice(x)
        return x


class Face_VGG_16(nn.Module):

    def __init__(self, require_grad=False):

        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        # self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        # self.fc7 = nn.Linear(4096, 4096)
        # self.fc8 = nn.Linear(4096, 2622)
        if not require_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        """ Pytorch forward

        Args:
            x: input image (224x224)

        Returns: class logits

        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        fea1 = x
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        fea2 = x
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        fea3 = x
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))

        # x = F.max_pool2d(x, 2, 2)
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc6(x))
        # x = F.dropout(x, 0.5)
        # x = F.relu(self.fc7(x))
        # x = F.dropout(x, 0.5)
        # x = self.fc8(x)
        return fea3


def global_feature_loss(model, pred, tagt, device):
    pred = (pred + 1) * 0.5 * 255
    tagt = (tagt + 1) * 0.5 * 255

    pred = F.adaptive_avg_pool2d(pred, (224, 224))
    tagt = F.adaptive_avg_pool2d(tagt, (224, 224))

    pred -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1).to(device)
    tagt -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1).to(device)
    pred_fea = model(pred)
    tagt_fea = model(tagt)
    return torch.mean((pred_fea - tagt_fea) ** 2)


def local_feature_loss(model, pred_, tagt_, device):
    # randomly crop 128x128 area from 512x512 images
    crop_h = np.random.randint(0, 512-128)
    crop_w = np.random.randint(0, 512-128)
    pred = pred_[:, :, crop_h: crop_h+128, crop_w: crop_w+128]
    tagt = tagt_[:, :, crop_h: crop_h+128, crop_w: crop_w+128]

    pred = (pred + 1) * 0.5
    tagt = (tagt + 1) * 0.5

    pred -= torch.Tensor(np.array([0.485, 0.456, 0.406])).view(1, 3, 1, 1).to(device)
    tagt -= torch.Tensor(np.array([0.485, 0.456, 0.406])).view(1, 3, 1, 1).to(device)
    pred /= torch.Tensor(np.array([0.229, 0.224, 0.225])).view(1, 3, 1, 1).to(device)
    tagt /= torch.Tensor(np.array([0.229, 0.224, 0.225])).view(1, 3, 1, 1).to(device)

    pred_fea = model(pred)
    tagt_fea = model(tagt)
    return torch.mean((pred_fea - tagt_fea) ** 2)


if __name__ == "__main__":
    model = Image_VGG_16()
    model.load_state_dict(torch.load("./weights/vgg/image_vgg.pth"), strict=True)
    print(model)
    # torch.save(model.state_dict(), "./image_vgg.pth")

    # model = VGG_16()
    # model.load_state_dict(torch.load("./weights/face_vgg_remove_top.pth"), strict=False)
    # torch.save(model.state_dict(), "./face_vgg_remove_top.pth")
    # print(torch.mean(model(x)))
