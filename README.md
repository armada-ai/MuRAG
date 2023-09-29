# MuRAG
## 1. setup environment
1. conda create n trans python=3.10
2. conda activate trans
3. pip install -r requirements.txt
## 2. run blip2&t5 model
python infer_vqa.py
<br/>note: ctrl+c for terminate
## 3. run rag_simple
python infer_rag_simple.py

## 4. run rag
python infer_rag.py

## 5. run vit
python infer_vit.py
