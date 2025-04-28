# from flask import Flask, render_template, request
# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch.nn.functional as F

# app = Flask(__name__)

# # Load BERT model
# model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# # Fungsi prediksi sentimen pakai BERT
# def predict_sentiment(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = F.softmax(outputs.logits, dim=-1)
#         pred_class = torch.argmax(probs, dim=1).item()

#         # Model ini punya label 0-4, di mana:
#         # 0: 1 star (buruk), 1: 2 star (buruk), 2: 3 star (netral), 3: 4 star (baik), 4: 5 star (baik)
#         if pred_class <= 1:
#             return 'Buruk'
#         elif pred_class == 2:
#             return 'Netral'
#         else:
#             return 'Baik'

# # Load data CSV saat server start
# try:
#     df = pd.read_csv('data/reviews_flip_2025.csv')
#     def classify_sentiment(score):
#         if score >= 4:
#             return 'Baik'
#         elif score == 3:
#             return 'Netral'
#         else:
#             return 'Buruk'

#     df['category'] = df['score'].apply(classify_sentiment)

#     gojek_df = pd.read_csv('data/reviews_gojek_2025.csv')
#     tokopedia_df = pd.read_csv('data/reviews_tokopedia_2025.csv')

#     gojek_df['category'] = gojek_df['score'].apply(classify_sentiment)
#     tokopedia_df['category'] = tokopedia_df['score'].apply(classify_sentiment)

# except Exception as e:
#     df = None
#     gojek_df = None
#     tokopedia_df = None
#     print(f"Error loading CSV: {e}")

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/reviews')
# def reviews():
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

#         reviews_data = []
#         for _, row in filtered_df.iterrows():
#             bert_category = predict_sentiment(row['content'])  # ← Tambahkan prediksi BERT di sini
#             reviews_data.append({
#                 'userName': row['userName'],
#                 'score': row['score'],
#                 'content': row['content'],
#                 'category_manual': row['category'],   # hasil dari skor manual
#                 'category_bert': bert_category         # hasil prediksi BERT
#             })

#         return render_template('reviews.html', reviews=reviews_data, search_query=search_query)
#     else:
#         return "Data tidak tersedia."

# # (bagian visualize tetap sama, kecuali kamu mau juga pakai prediksi BERT di sana)

# @app.route('/visualize')
# def visualize():
#     if df is not None:
#         if 'category' not in df.columns:
#             return "Kolom 'category' tidak ditemukan di data."

#         flip_counts = df['category'].value_counts().to_dict()

#         labels = list(flip_counts.keys())
#         flip_values = list(flip_counts.values())

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

#         return render_template('visualize.html',
#                                labels=labels,
#                                flip_values=flip_values,
#                                reviews=reviews_data,
#                                search_query=search_query)
#     else:
#         return "Data tidak tersedia."

# @app.route('/visualize_gojek')
# def visualize_gojek():
#     if gojek_df is not None:
#         if 'category' not in gojek_df.columns:
#             return "Kolom 'category' tidak ditemukan di data."

#         gojek_counts = gojek_df['category'].value_counts().to_dict()

#         labels = list(gojek_counts.keys())
#         gojek_values = list(gojek_counts.values())

#         search_query = request.args.get('search', '').lower()
#         filtered_df = gojek_df

#         if search_query:
#             filtered_df = gojek_df[
#                 gojek_df['userName'].str.lower().str.contains(search_query) |
#                 gojek_df['content'].str.lower().str.contains(search_query) |
#                 gojek_df['category'].str.lower().str.contains(search_query) |
#                 gojek_df['score'].astype(str).str.contains(search_query)
#             ]

#         reviews_data = filtered_df[['userName', 'score', 'content', 'category']].to_dict(orient='records')

#         return render_template('visualize_gojek.html',
#                                labels=labels,
#                                gojek_values=gojek_values,
#                                reviews=reviews_data,
#                                search_query=search_query)
#     else:
#         return "Data tidak tersedia."

# @app.route('/visualize_tokopedia')
# def visualize_tokopedia():
#     if tokopedia_df is not None:
#         if 'category' not in tokopedia_df.columns:
#             return "Kolom 'category' tidak ditemukan di data."

#         tokopedia_counts = tokopedia_df['category'].value_counts().to_dict()

#         labels = list(tokopedia_counts.keys())
#         tokopedia_values = list(tokopedia_counts.values())

#         search_query = request.args.get('search', '').lower()
#         filtered_df = tokopedia_df

#         if search_query:
#             filtered_df = tokopedia_df[
#                 tokopedia_df['userName'].str.lower().str.contains(search_query) |
#                 tokopedia_df['content'].str.lower().str.contains(search_query) |
#                 tokopedia_df['category'].str.lower().str.contains(search_query) |
#                 tokopedia_df['score'].astype(str).str.contains(search_query)
#             ]

#         reviews_data = filtered_df[['userName', 'score', 'content', 'category']].to_dict(orient='records')

#         return render_template('visualize_tokopedia.html',
#                                labels=labels,
#                                tokopedia_values=tokopedia_values,
#                                reviews=reviews_data,
#                                search_query=search_query)
#     else:
#         return "Data tidak tersedia."

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

app = Flask(__name__)

# Load BERT model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Fungsi prediksi sentimen pakai BERT
def predict_sentiment(text):
    if not text or pd.isna(text):  # Handle empty or NaN text
        return 'Tidak tersedia'
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=1).item()

            if pred_class <= 1:
                return 'Buruk'
            elif pred_class == 2:
                return 'Netral'
            else:
                return 'Baik'
    except Exception as e:
        print(f"Error predicting sentiment: {e}")
        return 'Error'

# Fungsi untuk memuat data dengan error handling
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"File {file_path} kosong")
            return None
            
        # Pastikan kolom yang diperlukan ada
        required_columns = ['userName', 'score', 'content']
        for col in required_columns:
            if col not in df.columns:
                print(f"Kolom {col} tidak ditemukan di {file_path}")
                return None
                
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Load data CSV saat server start dengan error handling
df = load_data('data/reviews_flip_2025.csv')
gojek_df = load_data('data/reviews_gojek_2025.csv')
tokopedia_df = load_data('data/reviews_tokopedia_2025.csv')

# Tambahkan kolom category jika data tersedia
def add_category_column(dataframe):
    if dataframe is not None:
        def classify_sentiment(score):
            if pd.isna(score):
                return 'Tidak tersedia'
            try:
                score = float(score)
                if score >= 4:
                    return 'Baik'
                elif score == 3:
                    return 'Netral'
                else:
                    return 'Buruk'
            except:
                return 'Error'
                
        dataframe['category'] = dataframe['score'].apply(classify_sentiment)
    return dataframe

df = add_category_column(df)
gojek_df = add_category_column(gojek_df)
tokopedia_df = add_category_column(tokopedia_df)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reviews')
def reviews():
    if df is None:
        return render_template('error.html', message="Data Flip tidak tersedia.")
    
    search_query = request.args.get('search', '').lower()
    filtered_df = df.copy() if df is not None else pd.DataFrame()

    if search_query and not filtered_df.empty:
        filtered_df = df[
            df['userName'].astype(str).str.lower().str.contains(search_query) |
            df['content'].astype(str).str.lower().str.contains(search_query) |
            df['category'].astype(str).str.lower().str.contains(search_query) |
            df['score'].astype(str).str.contains(search_query)
        ]

    reviews_data = []
    if not filtered_df.empty:
        for _, row in filtered_df.iterrows():
            bert_category = predict_sentiment(row['content'])
            reviews_data.append({
                'userName': row.get('userName', 'Tidak tersedia'),
                'score': row.get('score', 'Tidak tersedia'),
                'content': row.get('content', 'Tidak tersedia'),
                'category_manual': row.get('category', 'Tidak tersedia'),
                'category_bert': bert_category
            })

    return render_template('reviews.html', 
                          reviews=reviews_data, 
                          search_query=search_query,
                          data_available=df is not None)

@app.route('/visualize')
def visualize():
    if df is None:
        return render_template('error.html', message="Data Flip tidak tersedia.")
    
    if 'category' not in df.columns:
        return render_template('error.html', message="Kolom 'category' tidak ditemukan di data Flip.")

    flip_counts = df['category'].value_counts().to_dict()

    labels = list(flip_counts.keys())
    flip_values = list(flip_counts.values())

    search_query = request.args.get('search', '').lower()
    filtered_df = df.copy()

    if search_query:
        filtered_df = df[
            df['userName'].astype(str).str.lower().str.contains(search_query) |
            df['content'].astype(str).str.lower().str.contains(search_query) |
            df['category'].astype(str).str.lower().str.contains(search_query) |
            df['score'].astype(str).str.contains(search_query)
        ]

    reviews_data = filtered_df[['userName', 'score', 'content', 'category']].to_dict(orient='records')

    return render_template('visualize.html',
                         labels=labels,
                         flip_values=flip_values,
                         reviews=reviews_data,
                         search_query=search_query,
                         data_available=True)

@app.route('/visualize_gojek')
def visualize_gojek():
    if gojek_df is None:
        return render_template('error.html', message="Data Gojek tidak tersedia.")
    
    if 'category' not in gojek_df.columns:
        return render_template('error.html', message="Kolom 'category' tidak ditemukan di data Gojek.")

    gojek_counts = gojek_df['category'].value_counts().to_dict()

    labels = list(gojek_counts.keys())
    gojek_values = list(gojek_counts.values())

    search_query = request.args.get('search', '').lower()
    filtered_df = gojek_df.copy()

    if search_query:
        filtered_df = gojek_df[
            gojek_df['userName'].astype(str).str.lower().str.contains(search_query) |
            gojek_df['content'].astype(str).str.lower().str.contains(search_query) |
            gojek_df['category'].astype(str).str.lower().str.contains(search_query) |
            gojek_df['score'].astype(str).str.contains(search_query)
        ]

    reviews_data = filtered_df[['userName', 'score', 'content', 'category']].to_dict(orient='records')

    return render_template('visualize_gojek.html',
                         labels=labels,
                         gojek_values=gojek_values,
                         reviews=reviews_data,
                         search_query=search_query,
                         data_available=True)

@app.route('/visualize_tokopedia')
def visualize_tokopedia():
    if tokopedia_df is None:
        return render_template('error.html', message="Data Tokopedia tidak tersedia.")
    
    if 'category' not in tokopedia_df.columns:
        return render_template('error.html', message="Kolom 'category' tidak ditemukan di data Tokopedia.")

    tokopedia_counts = tokopedia_df['category'].value_counts().to_dict()

    labels = list(tokopedia_counts.keys())
    tokopedia_values = list(tokopedia_counts.values())

    search_query = request.args.get('search', '').lower()
    filtered_df = tokopedia_df.copy()

    if search_query:
        filtered_df = tokopedia_df[
            tokopedia_df['userName'].astype(str).str.lower().str.contains(search_query) |
            tokopedia_df['content'].astype(str).str.lower().str.contains(search_query) |
            tokopedia_df['category'].astype(str).str.lower().str.contains(search_query) |
            tokopedia_df['score'].astype(str).str.contains(search_query)
        ]

    reviews_data = filtered_df[['userName', 'score', 'content', 'category']].to_dict(orient='records')

    return render_template('visualize_tokopedia.html',
                         labels=labels,
                         tokopedia_values=tokopedia_values,
                         reviews=reviews_data,
                         search_query=search_query,
                         data_available=True)

if __name__ == '__main__':
    app.run(debug=True)