from transformers import pipeline, AutoModel, AutoTokenizer 
import torch


import requests
from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration

model_name = "ethzanalytics/blip2-flan-t5-xl-sharded"
processor = BlipProcessor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(model_name)

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
img_path = "/home/ubuntu/codes/MuRAG/sample20230919/images/train/chunk_7043_46.jpg"
raw_image = Image.open(img_path).convert("RGB")

while True:
    question = input("Question: ")
    inputs = processor(raw_image, question, return_tensors="pt")
    # import pdb
    # pdb.set_trace()
    out = model.generate(**inputs)
    print("Answer: ", processor.decode(out[0], skip_special_tokens=True))
