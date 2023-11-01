import cv2
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

top_k = 5
max_length = 70
max_length = 100
max_length = 150
max_length = 200
max_length = 300

vid_path = "755_1694792316.mp4"
reader = cv2.VideoCapture(vid_path)

width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = max(int(round(reader.get(cv2.CAP_PROP_FPS))), 1)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
save_vid_name = "%s_murag.mp4" % (os.path.basename(vid_path).split(".")[0])
print("save_vid_name: ", save_vid_name, os.path.dirname(save_vid_name))
writer = cv2.VideoWriter(save_vid_name, fourcc, fps, (width, height), True)

res_txts = []
count = 0
while True:
    ret, frame = reader.read()
    if not ret:
        break
    query_img_path = "tmp.png"
    cv2.imwrite(query_img_path, frame)
    query_outputs = get_query_outputs(query_img_path, processor, model)
    if os.path.exists(query_img_path):
        os.remove(query_img_path)
    count += 1

    searched_caption = search(query_outputs, caches, top_k)
    print("[QUERY_IMG_INFO]: frame %s" % (count))
    print("[SEARCHED_CAPTION(frame %s)]: %s" % (count, searched_caption))
    
    out = summarizer(searched_caption, max_length=max_length)
    final_caption = out[0]["summary_text"]
    print("[FINAL_CAPTION(frame %s)]: %s" % (count, final_caption))
    print("---------------------seperator----------------------\n")
    # break
    # Process the result if available (None if no result yet)
    color = (0, 0, 180)
    if final_caption is not None:
        res_text = "[MURAG]frame: %d; res: %s" % (count, final_caption)
        cv2.putText(frame, res_text, (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        res_txts.append(res_text)
        writer.write(frame)

reader.release()
writer.release()
    
save_txt_name = "%s_murag.txt" % (os.path.basename(vid_path).split(".")[0])
print("save_txt_name: ", save_txt_name, os.path.dirname(save_txt_name))
with open(save_txt_name, "w", encoding="utf-8") as f:
        f.write("\n".join(res_txts))



