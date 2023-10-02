from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration 
 
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq") 
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True) 
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever) 

question = "how many countries are in europe"
input_dict = tokenizer.prepare_seq2seq_batch(question, return_tensors="pt") 

generated = model.generate(input_ids=input_dict["input_ids"]) 
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0]) 