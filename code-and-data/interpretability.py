import torch
from torch import nn
from torch import optim
from transformer import TransformerLM
import data
import lm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "data/"

tokenizer, tokenized_data = data.load_data(data_path)
pad_id = tokenizer.pad_id()

model_path = "transformer_lm.pth"
model = TransformerLM.load_model(model_path, device)
print("Model loaded from checkpoint.")


model.eval()
sampled = tokenizer.detokenize(model.better_sample_continuation(tokenizer.tokenize("Hello"), 500, temperature=0.7, topK=5))
print(f"Model sample: '''{sampled}'''")