from transformers import BertForSequenceClassification, BertTokenizer

import torch
# Inisialisasi model BERT untuk klasifikasi
model = BertForSequenceClassification.from_pretrained(
    "indobenchmark/indobert-base-p1"  # Ganti dengan path model Anda jika custom
)
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

# Muat weights dari file .pth
state_dict = torch.load("model/sentiment_model.pth")
model.load_state_dict(state_dict)

# Pindahkan ke device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)