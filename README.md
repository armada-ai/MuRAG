# MuRAG

## 1. setup environment

1. conda create -n trans python=3.10
2. conda activate trans
3. pip install -r requirements.txt

## 2. run blip2&t5 model

python infer_vqa.py
**Note: ctrl+c for terminate.** If you want to test other images, please change the `img_path` in infer_vqa.py.

## 3. run rag_simple model

python infer_rag_simple.py

You can change `search_img_root` to change to your own database. 

You can upload your images to `query_img_root` to test. Besides, you can also change `query_img_root` to the directory where test images are located.

## 4. run rag model

python infer_rag.py

You can change `question` to use your own question.

# 5. Explanations of these models

## 5.1 blip2&t5 model

This model is used to complete visual question answering (VQA) tasks. 
Input:  

1) the image path(`img_path`)
2) the question(`question`)

Output:

1. the answer of the question

The model uses blip2 to get the embedding of the image, uses T5 to extract the embedding of the question, then concats the two embeddings, and inputs it into the subsequent module to generate the answer.

## 5.2 rag_simple model

This model is used to complete visual question answering (VQA) tasks simply. 
Input:  

1) the image root(`query_img_root`)

Output:

1. the caption of each image in the image root(`query_img_root`)

This model only contains the retrieval stage. First, this model uses vit to cache all image embeddings and corresponding captions under the specified data set path(`search_img_root`). Then specify the test image path(`query_img_root`), for each image of this root: 

1. use vit to extract the embedding of the image, 
2. calculate the inner product with all the image emeddings in the database
3. obtain the caption of the image with the largest inner product, 
4. use this caption as the input image caption

## 5.3 rag model

This model is used to complete question answering (QA) tasks simply. 
Input:  

1) the question(`question`)

Output:

1. the answer of the question

The model contains a retriever and a generator. The former consists of a question encoder and a document index, which are responsible for question encoding and document indexing respectively; the latter is a seq2seq generative model. In the retrieval stage, the maximum inner product search method (MIPS) is used to retrieve top-K related documents. For more info, please refer to [RAG](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)