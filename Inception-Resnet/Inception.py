import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import dataset as ds
import classification as cls


if __name__ == "__main__":
    # get transformed data
    dataset = ds.loadImages('C:/Users/yifan/Desktop/Intro_AI/CT')

    # define a pretrained model
    mobiles_v2_model = models.mobilenet_v2(pretrained=True)
    print(mobiles_v2_model)

    mobiles_v2_model.eval()