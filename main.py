import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
model.eval()  # change model's mode to eval

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img)  # image to tensor
    return img_tensor

def predict(model, img_tensor):
    with torch.no_grad():  # don't calculate gradient while prediction
        prediction = model([img_tensor.to(device)])
    return prediction[0]

def plot_image_with_boxes(img, predictions):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(np.array(img))

    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    for box, score in zip(boxes, scores):
        if score > 0.8: # high score show box
            x_min, y_min, x_max, y_max = box
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color="red", linewidth=2)
            ax.add_patch(rect)
            ax.text(x_min, y_min - 10, f"{score:.2f}", color="red", fontsize=12)

    plt.axis("off")
    plt.show()

device = torch.device("cpu")
model.to(device)

image_path = "image.png"
img_tensor = load_image(image_path)
predictions = predict(model, img_tensor)

img = Image.open(image_path).convert("RGB")
plot_image_with_boxes(img, predictions)
