from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)



from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")



## Get Image Features ##
inputs = processor(images=image, return_tensors="pt")
image_features = model.clip.get_image_features(**inputs)
########################


_out = model.clip.vision_model(**inputs, output_hidden_states=True)
print(_out['hidden_states'][-1].shape)



prompts = ["a box", "a blue cube", "white", "a cube", "a very very very long text is going here"]

inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")

# predict
with torch.no_grad():
  outputs = model(**inputs)

preds = outputs.logits.unsqueeze(1)

# visualize prediction
_, ax = plt.subplots(1, 5, figsize=(15, 4))
[a.axis('off') for a in ax.flatten()]
ax[0].imshow(image)
[ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(4)];
[ax[i+1].text(0, -15, prompts[i]) for i in range(4)];
plt.show()