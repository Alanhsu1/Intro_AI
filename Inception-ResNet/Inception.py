import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import dataset as ds
import classification as cls
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PyTorch Image Classification")
    parser.add_argument("--image_path", type=str,
                        default="images/maltese.jpeg", help="path to image")
    parser.add_argument("--class_def", type=str,
                        default="imagenet_classes.txt", help="path to ImageNet class definition")

    # Parse arguments
    args = parser.parse_args()

    print(dir(models))

    # Load ImageNet classes
    with open(args.class_def) as f:
        classes = [line.strip() for line in f.readlines()]

    # get transformed data
    dataset = ds.loadImages('C:/Users/yifan/Desktop/Intro_AI/CT')

    # define a pretrained model
    mobiles_v2_model = models.mobilenet_v2(pretrained=True)
    print(mobiles_v2_model)

    mobiles_v2_model.eval()

    with torch.no_grad():
        cls.Image_Classification(mobiles_v2_model, dataset, classes)    