import torch
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")








