import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as trans
from PIL import Image

def Image_Classification(model, dataset, transform, classes, topk=5):
    
    image = dataset
    
