from transformers import pipeline

summarizer = pipeline(task="summarization")#, model="t5-base", tokenizer="t5-base")
input_text = "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles."
out = summarizer(input_text, max_length=70)#, min_length=40)
# print(out)
summary_text = out[0]["summary_text"]
print("[INPUT_TEXT]: ", input_text)
print("[SUMMARY_TEXT]: ", summary_text)
