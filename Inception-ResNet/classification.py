import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as trans
from PIL import Image

def Image_Classification(model, dataset, classes, topk=5):
    
    image = dataset

    for file in dataset:
        # feed input
        output = model(file)
        print(f"Output size: {output.size()}")

        output = output.squeeze()
        print(f"Output size after squeezing: {output.size()}")
    
        _, indices = torch.sort(output, descending=True)
        probs = F.softmax(output, dim=-1)

        print("\n\n Inference results: ")
        for index in indices[:topk]:
            print(f"Label {index}: {classes[index]} ({probs[index].item():.2f})")

    return

