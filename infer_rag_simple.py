import os
import json
import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from glob import glob
from tqdm import tqdm


def cache_dataset_embeds(search_img_root, processor, model):
    cache_save_path = "sample20230919_embeds.dict"
    if os.path.exists(cache_save_path):
        caches = torch.load(cache_save_path)
    else:
        with open("/home/ubuntu/codes/MuRAG/sample20230919/annotations/sample048_train.json", encoding="utf-8") as f:
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
            caches[img_name]["query_last_hidden_states"] = outputs.last_hidden_state.squeeze()
            caches[img_name]["pooler_output"] = outputs.pooler_output.squeeze()
            caches[img_name]["caption"] = captions[img_name]
        if len(caches) > 0:
            torch.save(caches, cache_save_path)
    return caches


def get_query_outputs(query_img_path, processor, model):
    query_image = Image.open(query_img_path)
    query_inputs = processor(images=query_image, return_tensors="pt")
    query_outputs = model(**query_inputs)
    query_last_hidden_states = query_outputs.last_hidden_state
    query_pooler_output = query_outputs.pooler_output
    return {"query_last_hidden_states": query_last_hidden_states, "query_pooler_output": query_pooler_output}


def search(query_outputs, dataset_infos):
    max_inner_product = -10
    # max_img_name = ""
    max_caption = ""

    for img_name in dataset_infos.keys():
        query_last_hidden_states = query_outputs["query_last_hidden_states"]
        img_infos = dataset_infos[img_name]

        dataset_last_hidden_states = img_infos["query_last_hidden_states"]
        caption = img_infos["caption"]

        inner_product_sum = torch.mul(query_last_hidden_states, dataset_last_hidden_states).sum()
        if max_inner_product <= inner_product_sum:
            max_inner_product = inner_product_sum
            # max_img_name = img_name
            max_caption = caption


    # print("inner_product_sum: ", max_inner_product.detach().cpu().numpy())
    # print("img_name: ", max_img_name)
    # print("caption: ", max_caption)
    return max_caption


processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')




search_img_root = "/home/ubuntu/codes/MuRAG/sample20230919/images/train"
caches = cache_dataset_embeds(search_img_root, processor, model)

query_img_root = "/home/ubuntu/codes/MuRAG/sample20230919/images/test"
query_img_paths = glob("%s/*" % (query_img_root))
query_img_paths = sorted(query_img_paths)
for query_img_path in query_img_paths:
    # query_img_path = "/home/ubuntu/codes/MuRAG/sample20230919/images/test/chunk_7044_13.jpg"
    query_outputs = get_query_outputs(query_img_path, processor, model)

    caption = search(query_outputs, caches)
    print("query_img_path: ", query_img_path)
    print("caption: ", caption)
    print("---------------------seperator---------------")



