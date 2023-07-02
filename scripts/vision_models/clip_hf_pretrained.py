from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)



### Get Latent Text Features ####
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
text_features = model.get_text_features(**inputs)
#########

## Get Latent Image Features ##
inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**inputs)
###############################


_out = model.vision_model(pixel_values=inputs['pixel_values'],
                output_attentions=None,
                output_hidden_states=True,  # we need the intermediate hidden states
                return_dict=True,)


inputs = processor(text=["a photo of a cat", "a photo of a dog", "cat", "two cats in the sofa", "two cats in the sofa and two tv controllers"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)