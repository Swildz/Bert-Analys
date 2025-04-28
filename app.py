from flask import Flask, render_template, request
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(__name__)

# Load pretrained model and tokenizer
MODEL_NAME = 'nlptown/bert-base-multilingual-uncased-sentiment'

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()  # Set model ke eval mode
    print("✅ Pretrained model berhasil dimuat.")
except Exception as e:
    tokenizer, model = None, None
    print(f"❌ Error loading model: {e}")

# Fungsi untuk klasifikasi berdasarkan skor numerik
def classify_sentiment(score):
    if score >= 4:
        return 'Baik'
    elif score == 3:
        return 'Netral'
    else:
        return 'Buruk'

# Fungsi untuk klasifikasi menggunakan model
def classify_sentiment_model(text):
    if not model or not tokenizer:
        return "Model tidak tersedia"

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item() + 1  # Kelas mulai dari 0, tambah 1

        if pred_class >= 4:
            return 'Baik'
        elif pred_class == 3:
            return 'Netral'
        else:
            return 'Buruk'
    except Exception as e:
        print(f"Error saat klasifikasi: {e}")
        return "Terjadi kesalahan dalam klasifikasi"

# Load datasets
def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        data['category_score'] = data['score'].apply(classify_sentiment)
        data['category_model'] = data['content'].apply(classify_sentiment_model)
        return data
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

df = load_data('data/reviews_flip_2025.csv')
gojek_df = load_data('data/reviews_gojek_2025.csv')
tokopedia_df = load_data('data/reviews_tokopedia_2025.csv')

# Fungsi general untuk visualisasi
def visualize_data(dataframe, template_name):
    if dataframe is None:
        return "Data tidak tersedia."

    method = request.args.get('method', 'score')
    category_column = 'category_score' if method == 'score' else 'category_model'

    search_query = request.args.get('search', '').lower()
    filtered_df = dataframe

    if search_query:
        filtered_df = dataframe[
            dataframe['userName'].str.lower().str.contains(search_query) |
            dataframe['content'].str.lower().str.contains(search_query) |
            dataframe[category_column].str.lower().str.contains(search_query) |
            dataframe['score'].astype(str).str.contains(search_query)
        ]

    counts = filtered_df[category_column].value_counts().to_dict()
    labels = list(counts.keys())
    values = list(counts.values())

    reviews_data = filtered_df[['userName', 'score', 'content', category_column]]
    reviews_data = reviews_data.rename(columns={category_column: 'category'}).to_dict(orient='records')

    return render_template(
        template_name,
        labels=labels,
        values=values,
        reviews=reviews_data,
        search_query=search_query,
        method=method
    )

# Routes
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
