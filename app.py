from flask import Flask, render_template, request
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

app = Flask(__name__)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load BERT model and move to device (GPU if available)
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# Fungsi prediksi sentimen pakai BERT
def predict_sentiment(text):
    if not text or pd.isna(text):  # Handle empty or NaN text
        return 'Tidak tersedia'
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        # Move inputs to the same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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

# Batch sentiment prediction to make better use of GPU
def batch_predict_sentiments(texts, batch_size=16):
    results = []
    
    # Handle None or empty list
    if not texts or len(texts) == 0:
        return []
    
    # Filter out None or NaN texts
    valid_texts = [t for t in texts if t and not pd.isna(t)]
    invalid_indices = [i for i, t in enumerate(texts) if not t or pd.isna(t)]
    
    for i in range(0, len(valid_texts), batch_size):
        batch_texts = valid_texts[i:i+batch_size]
        try:
            # Tokenize batch
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                pred_classes = torch.argmax(probs, dim=1).tolist()
            
            # Map to sentiment labels
            batch_sentiments = []
            for cls in pred_classes:
                if cls <= 1:
                    batch_sentiments.append('Buruk')
                elif cls == 2:
                    batch_sentiments.append('Netral')
                else:
                    batch_sentiments.append('Baik')
                    
            results.extend(batch_sentiments)
        except Exception as e:
            print(f"Error in batch prediction: {e}")
            # If batch fails, fall back to individual prediction
            batch_results = [predict_sentiment(t) for t in batch_texts]
            results.extend(batch_results)
    
    # Reinsert 'Tidak tersedia' for invalid texts
    final_results = []
    valid_idx = 0
    for i in range(len(texts)):
        if i in invalid_indices:
            final_results.append('Tidak tersedia')
        else:
            if valid_idx < len(results):
                final_results.append(results[valid_idx])
                valid_idx += 1
            else:
                final_results.append('Error')
    
    return final_results

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
        
        # Add BERT sentiment predictions (batch processing for efficiency)
        if 'content' in dataframe.columns:
            content_list = dataframe['content'].tolist()
            # Use batched prediction for better GPU utilization
            bert_predictions = batch_predict_sentiments(content_list)
            dataframe['category_bert'] = bert_predictions
    return dataframe

# Process all dataframes
print("Processing Flip data...")
df = add_category_column(df)
print("Processing Gojek data...")
gojek_df = add_category_column(gojek_df)
print("Processing Tokopedia data...")
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
            df['category_bert'].astype(str).str.lower().str.contains(search_query) |
            df['score'].astype(str).str.contains(search_query)
        ]

    reviews_data = []
    if not filtered_df.empty:
        for _, row in filtered_df.iterrows():
            reviews_data.append({
                'userName': row.get('userName', 'Tidak tersedia'),
                'score': row.get('score', 'Tidak tersedia'),
                'content': row.get('content', 'Tidak tersedia'),
                'category': row.get('category', 'Tidak tersedia'),
                'category_bert': row.get('category_bert', 'Tidak tersedia')
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
    
    if 'category_bert' not in df.columns:
        return render_template('error.html', message="Kolom 'category_bert' tidak ditemukan di data Flip.")

    # Get manual sentiment counts
    flip_counts = df['category'].value_counts().to_dict()
    # Get BERT sentiment counts
    flip_bert_counts = df['category_bert'].value_counts().to_dict()

    labels = list(flip_counts.keys())
    bert_labels = list(flip_bert_counts.keys()) 
    flip_values = list(flip_counts.values())
    flip_bert_values = list(flip_bert_counts.values())

    search_query = request.args.get('search', '').lower()
    filtered_df = df.copy()

    if search_query:
        filtered_df = df[
            df['userName'].astype(str).str.lower().str.contains(search_query) |
            df['content'].astype(str).str.lower().str.contains(search_query) |
            df['category'].astype(str).str.lower().str.contains(search_query) |
            df['category_bert'].astype(str).str.lower().str.contains(search_query) |
            df['score'].astype(str).str.contains(search_query)
        ]

    reviews_data = filtered_df[['userName', 'score', 'content', 'category', 'category_bert']].to_dict(orient='records')

    return render_template('visualize.html',
                         labels=labels,
                         bert_labels=bert_labels,
                         flip_values=flip_values,
                         flip_bert_values=flip_bert_values,
                         reviews=reviews_data,
                         search_query=search_query,
                         data_available=True)

@app.route('/visualize_gojek')
def visualize_gojek():
    if gojek_df is None:
        return render_template('error.html', message="Data Gojek tidak tersedia.")
    
    if 'category' not in gojek_df.columns:
        return render_template('error.html', message="Kolom 'category' tidak ditemukan di data Gojek.")
    
    if 'category_bert' not in gojek_df.columns:
        return render_template('error.html', message="Kolom 'category_bert' tidak ditemukan di data Gojek.")

    # Get manual sentiment counts
    gojek_counts = gojek_df['category'].value_counts().to_dict()
    # Get BERT sentiment counts
    gojek_bert_counts = gojek_df['category_bert'].value_counts().to_dict()

    labels = list(gojek_counts.keys())
    bert_labels = list(gojek_bert_counts.keys())
    gojek_values = list(gojek_counts.values())
    gojek_bert_values = list(gojek_bert_counts.values())

    search_query = request.args.get('search', '').lower()
    filtered_df = gojek_df.copy()

    if search_query:
        filtered_df = gojek_df[
            gojek_df['userName'].astype(str).str.lower().str.contains(search_query) |
            gojek_df['content'].astype(str).str.lower().str.contains(search_query) |
            gojek_df['category'].astype(str).str.lower().str.contains(search_query) |
            gojek_df['category_bert'].astype(str).str.lower().str.contains(search_query) |
            gojek_df['score'].astype(str).str.contains(search_query)
        ]

    reviews_data = filtered_df[['userName', 'score', 'content', 'category', 'category_bert']].to_dict(orient='records')

    return render_template('visualize_gojek.html',
                         labels=labels,
                         bert_labels=bert_labels,
                         gojek_values=gojek_values,
                         gojek_bert_values=gojek_bert_values,
                         reviews=reviews_data,
                         search_query=search_query,
                         data_available=True)

@app.route('/visualize_tokopedia')
def visualize_tokopedia():
    if tokopedia_df is None:
        return render_template('error.html', message="Data Tokopedia tidak tersedia.")
    
    if 'category' not in tokopedia_df.columns:
        return render_template('error.html', message="Kolom 'category' tidak ditemukan di data Tokopedia.")
    
    if 'category_bert' not in tokopedia_df.columns:
        return render_template('error.html', message="Kolom 'category_bert' tidak ditemukan di data Tokopedia.")

    # Get manual sentiment counts
    tokopedia_counts = tokopedia_df['category'].value_counts().to_dict()
    # Get BERT sentiment counts
    tokopedia_bert_counts = tokopedia_df['category_bert'].value_counts().to_dict()

    labels = list(tokopedia_counts.keys())
    bert_labels = list(tokopedia_bert_counts.keys())
    tokopedia_values = list(tokopedia_counts.values())
    tokopedia_bert_values = list(tokopedia_bert_counts.values())

    search_query = request.args.get('search', '').lower()
    filtered_df = tokopedia_df.copy()

    if search_query:
        filtered_df = tokopedia_df[
            tokopedia_df['userName'].astype(str).str.lower().str.contains(search_query) |
            tokopedia_df['content'].astype(str).str.lower().str.contains(search_query) |
            tokopedia_df['category'].astype(str).str.lower().str.contains(search_query) |
            tokopedia_df['category_bert'].astype(str).str.lower().str.contains(search_query) |
            tokopedia_df['score'].astype(str).str.contains(search_query)
        ]

    reviews_data = filtered_df[['userName', 'score', 'content', 'category', 'category_bert']].to_dict(orient='records')

    return render_template('visualize_tokopedia.html',
                         labels=labels,
                         bert_labels=bert_labels,
                         tokopedia_values=tokopedia_values,
                         tokopedia_bert_values=tokopedia_bert_values,
                         reviews=reviews_data,
                         search_query=search_query,
                         data_available=True)

if __name__ == '__main__':
    app.run(debug=True)