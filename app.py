# # from flask import Flask, render_template, request
# # import pandas as pd

# # app = Flask(__name__)

# # # Load data CSV saat server start
# # try:
# #     df = pd.read_csv('data/reviews_flip_2025.csv')  # Ganti nama file sesuai kebutuhan
# #     # Tambahkan kolom category berdasarkan score
# #     def classify_sentiment(score):
# #         if score >= 4:
# #             return 'Baik'
# #         elif score == 3:
# #             return 'Netral'
# #         else:
# #             return 'Buruk'

# #     df['category'] = df['score'].apply(classify_sentiment)

# # except Exception as e:
# #     df = None
# #     print(f"Error loading CSV: {e}")

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # @app.route('/reviews')
# # def reviews():
# #     if df is not None:
# #         search_query = request.args.get('search', '').lower()
# #         filtered_df = df

# #         if search_query:
# #             filtered_df = df[
# #                 df['userName'].str.lower().str.contains(search_query) |
# #                 df['content'].str.lower().str.contains(search_query) |
# #                 df['category'].str.lower().str.contains(search_query) |
# #                 df['score'].astype(str).str.contains(search_query)
# #             ]

# #         reviews_data = filtered_df[['userName', 'score', 'content', 'category']].to_dict(orient='records')
# #         return render_template('reviews.html', reviews=reviews_data, search_query=search_query)
# #     else:
# #         return "Data tidak tersedia."

# # @app.route('/visualize')
# # def visualize():
# #     if df is not None:
# #         if 'category' not in df.columns:
# #             return "Kolom 'category' tidak ditemukan di data."

# #         # Hitung jumlah setiap kategori
# #         category_counts = df['category'].value_counts().to_dict()
# #         labels = list(category_counts.keys())
# #         values = list(category_counts.values())

# #         return render_template('visualize.html', labels=labels, values=values)
# #     else:
# #         return "Data tidak tersedia."

# # if __name__ == '__main__':
# #     app.run(debug=True)

# from flask import Flask, render_template, request
# import pandas as pd

# app = Flask(__name__)

# # Load semua data saat server start
# data_sources = {
#     'flip': 'data/reviews_flip_2025.csv',
#     'shopee': 'data/reviews_shopee_2025.csv',
#     'tokped': 'data/reviews_tokped_2025.csv'
# }

# dfs = {}

# # Load semua CSV
# for key, path in data_sources.items():
#     try:
#         df = pd.read_csv(path)
#         def classify_sentiment(score):
#             if score >= 4:
#                 return 'Baik'
#             elif score == 3:
#                 return 'Netral'
#             else:
#                 return 'Buruk'
#         df['category'] = df['score'].apply(classify_sentiment)
#         dfs[key] = df
#     except Exception as e:
#         print(f"Error loading {key}: {e}")

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/reviews')
# def reviews():
#     platform = request.args.get('platform', 'flip')  # default flip
#     df = dfs.get(platform)

#     if df is not None:
#         search_query = request.args.get('search', '').lower()
#         filtered_df = df

#         if search_query:
#             filtered_df = df[
#                 df['userName'].str.lower().str.contains(search_query) |
#                 df['content'].str.lower().str.contains(search_query) |
#                 df['category'].str.lower().str.contains(search_query) |
#                 df['score'].astype(str).str.contains(search_query)
#             ]

#         reviews_data = filtered_df[['userName', 'score', 'content', 'category']].to_dict(orient='records')
#         return render_template('reviews.html', reviews=reviews_data, search_query=search_query, platform=platform)
#     else:
#         return "Data tidak tersedia."

# @app.route('/visualize')
# def visualize():
#     platform = request.args.get('platform', 'flip')
#     df = dfs.get(platform)

#     if df is not None:
#         if 'category' not in df.columns:
#             return "Kolom 'category' tidak ditemukan di data."

#         category_counts = df['category'].value_counts().to_dict()
#         labels = list(category_counts.keys())
#         values = list(category_counts.values())

#         return render_template('visualize.html', labels=labels, values=values, platform=platform)
#     else:
#         return "Data tidak tersedia."

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = '212192919291eu2eu1ueu'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Load model (sesuaikan dengan arsitektur model Anda)
class SentimentClassifier(nn.Module):
    def __init__(self, model_name='indobenchmark/indobert-base-p1', num_labels=3):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits

# Inisialisasi model
model = SentimentClassifier()
model.load_state_dict(torch.load('model/sentiment_model.pth', map_location=torch.device('cpu')))
model.eval()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

# Fungsi untuk memeriksa ekstensi file
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Fungsi untuk analisis sentimen
def analyze_sentiment(texts):
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.argmax(outputs, dim=-1).item()
        
        # Map prediction to label (sesuaikan dengan model Anda)
        sentiment_labels = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
        sentiment = sentiment_labels.get(prediction, 'Tidak Diketahui')
        
        results.append({
            'text': text,
            'sentiment': sentiment,
            'prediction': prediction
        })
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        flash('Tidak ada file yang diunggah')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('Tidak ada file yang dipilih')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        try:
            # Baca file CSV
            df = pd.read_csv(filepath)
            if 'text' not in df.columns:
                flash('File CSV harus memiliki kolom "text"')
                return redirect(url_for('index'))
            
            texts = df['text'].astype(str).tolist()
            results = analyze_sentiment(texts[:1000])  # Batasi 1000 teks untuk demo
            
            # Simpan hasil ke DataFrame
            results_df = pd.DataFrame(results)
            results_csv = results_df.to_csv(index=False)
            
            return render_template('results.html', 
                                 results=results,
                                 results_csv=results_csv,
                                 filename=filename)
        
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Format file tidak didukung. Harap unggah file CSV.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)