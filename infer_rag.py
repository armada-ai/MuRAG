from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, pipeline
import torch


tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True
)
# initialize with RagRetriever to do everything in one forward call
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")
# print("rag_infer model: ", model)

question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer)
question = "Where do I work?"
context = "My name is Sylvain and I work at Hugging Face in Brooklyn"
answer = question_answerer(
    question=question,
    context=context,
)
print("Q: ", question)
print("Context: ", context)
print("answer: ", answer)


# inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
# targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
# input_ids = inputs["input_ids"]
# labels = targets["input_ids"]
# outputs = model(input_ids=input_ids, labels=labels)
# import pdb
# pdb.set_trace()

# # or use retriever separately
# model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)
# # 1. Encode
# question_hidden_states = model.question_encoder(input_ids)[0]
# # 2. Retrieve
# docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
# doc_scores = torch.bmm(
#     question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
# ).squeeze(1)
# # 3. Forward to generator
# outputs = model(
#     context_input_ids=docs_dict["context_input_ids"],
#     context_attention_mask=docs_dict["context_attention_mask"],
#     doc_scores=doc_scores,
#     decoder_input_ids=labels,
# )

# import pdb
# pdb.set_trace()
# print("outputs: ", outputs.keys())




