import os
import json
import torch
from PIL import Image
from glob import glob
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel, pipeline


def cache_dataset_embeds(search_img_root, processor, model):
    cache_save_path = "sample20230919_embeds.dict"
    if os.path.exists(cache_save_path):
        caches = torch.load(cache_save_path)
    else:
        with open("./sample20230919/annotations/sample048_train.json", encoding="utf-8") as f:
            datas = json.load(f)
            
        # import pdb
        # pdb.set_trace()
        
        captions = {}
        for data in datas:
            img_rel_path = data["image"]
            img_name = img_rel_path.split("/")[-1].split(".")[0]
            captions[img_name] = data["caption"]


        img_paths = glob("%s/*" % (search_img_root))
        
        # import pdb
        # pdb.set_trace()
        
        caches = {}
        for img_path in tqdm(img_paths):
            image = Image.open(img_path)
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            img_name = os.path.basename(img_path).split(".")[0]
            caches[img_name] = {}
            caches[img_name]["last_hidden_states"] = outputs.last_hidden_state.squeeze()
            caches[img_name]["pooler_output"] = outputs.pooler_output.squeeze()
            caches[img_name]["caption"] = captions[img_name]
        if len(caches) > 0:
            torch.save(caches, cache_save_path)
    return caches


def get_query_outputs(query_img_path, processor, model):
    query_image = Image.open(query_img_path)
    query_inputs = processor(images=query_image, return_tensors="pt")
    query_outputs = model(**query_inputs)
    query_last_hidden_states = query_outputs.last_hidden_state.squeeze()
    query_pooler_output = query_outputs.pooler_output.squeeze()
    return {"last_hidden_states": query_last_hidden_states, "pooler_output": query_pooler_output}


def search(query_outputs, dataset_infos, top_k=5):
    query_last_hidden_states = query_outputs["pooler_output"]
    device = query_last_hidden_states.device
    dataset_embeddings = []
    dataset_captions = []
    for img_name in dataset_infos.keys():
        img_infos = dataset_infos[img_name]
        dataset_last_hidden_states = img_infos["pooler_output"]
        caption = img_infos["caption"]
        if len(dataset_last_hidden_states.shape) == 1:
            dataset_last_hidden_states = dataset_last_hidden_states.unsqueeze(0)
        dataset_embeddings.append(dataset_last_hidden_states)
        dataset_captions.append(caption)
    
    # import pdb
    # pdb.set_trace()
    
    dataset_embeddings = torch.cat(dataset_embeddings, axis=0)
    dataset_embeddings.to(device)
    
    if len(query_last_hidden_states.shape) == 1:
        query_last_hidden_states = query_last_hidden_states.unsqueeze(0)
    
    inner_product_sum = torch.mul(query_last_hidden_states, dataset_embeddings).sum(1)
    idx = torch.argsort(inner_product_sum, descending=True)
    idx = idx.detach().cpu().numpy()
    # print("top-k idxs: ", idx[: top_k])
    caption = ""
    for i in range(top_k):
        caption = "%s %s" % (caption, dataset_captions[idx[i]])
    
    return caption[1:]


processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
summarizer = pipeline(task="summarization", model="t5-base", tokenizer="t5-base")



search_img_root = "./sample20230919/images/train"
caches = cache_dataset_embeds(search_img_root, processor, model)

query_img_root = "./sample20230919/images/test"
query_img_paths = glob("%s/*" % (query_img_root))
query_img_paths = sorted(query_img_paths)

top_k = 5
max_length = 70
max_length = 100
max_length = 150
max_length = 200
max_length = 300

for query_img_path in query_img_paths:
    # query_img_path = "/home/ubuntu/codes/MuRAG/sample20230919/images/test/chunk_7044_13.jpg"
    query_outputs = get_query_outputs(query_img_path, processor, model)

    searched_caption = search(query_outputs, caches, top_k)
    print("[QUERY_IMG_PATH]: ", query_img_path)
    print("[SEARCHED_CAPTION]: ", searched_caption)
    
    out = summarizer(searched_caption, max_length=max_length)
    final_caption = out[0]["summary_text"]
    print("[FINAL_CAPTION]: ", final_caption)
    print("---------------------seperator----------------------\n")
    # break



