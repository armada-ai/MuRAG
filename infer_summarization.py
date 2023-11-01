from transformers import pipeline

summarizer = pipeline(task="summarization")#, model="t5-base", tokenizer="t5-base")
input_text = "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles."
with open("755_1694792316_murag.txt") as f:
    lines = f.readlines()
input_text = ""
text = []
for line in lines:
    text.append(line.strip().split(": ")[-1])
input_text = " ".join(list(set(text)))
print("input_text: ", input_text)
out = summarizer(input_text[: 1000], max_length=100)#, min_length=40)
# print(out)
summary_text = out[0]["summary_text"]
print("[INPUT_TEXT]: ", input_text)
print("[SUMMARY_TEXT]: ", summary_text)
