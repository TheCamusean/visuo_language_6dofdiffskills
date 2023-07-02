from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import requests
import torch

# Load Mask2Former trained on CityScapes panoptic segmentation dataset
image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-small-cityscapes-panoptic"
)

url = "https://cdn-media.huggingface.co/Inference-API/Sample-results-on-the-Cityscapes-dataset-The-above-images-show-how-our-method-can-handle.png"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# Perform post-processing to get panoptic segmentation map
pred_panoptic_map = image_processor.post_process_panoptic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]["segmentation"]
print(pred_panoptic_map.shape)

### Visualize Image ###
import matplotlib.pyplot as plt

img = inputs['pixel_values'][0,...].permute(1,2,0).numpy()
mask = inputs['pixel_mask'][0,...].numpy()

plt.imshow(pred_panoptic_map.numpy())
plt.show()


