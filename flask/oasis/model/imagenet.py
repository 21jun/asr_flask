
import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image


class Densenet121:

    def __init__(self):
        self.imagenet_class_index = json.load(
            open('oasis/model/imagenet_class_index.json'))
        self.model = models.densenet121(pretrained=True)
        self.model.eval()

    def transform_image(self, image_bytes):
        my_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                [0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        return my_transforms(image).unsqueeze(0)

    def get_prediction(self, image_bytes):
        tensor = self.transform_image(image_bytes=image_bytes)
        outputs = self.model.forward(tensor)
        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())
        return self.imagenet_class_index[predicted_idx]
