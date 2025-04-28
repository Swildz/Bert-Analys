from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load data CSV saat server start
try:
    # Load data utama
    df = pd.read_csv('data/reviews_flip_2025.csv')  # Ganti nama file sesuai kebutuhan
    # Tambahkan kolom category berdasarkan score
    def classify_sentiment(score):
        if score >= 4:
            return 'Baik'
        elif score == 3:
            return 'Netral'
        else:
            return 'Buruk'

    df['category'] = df['score'].apply(classify_sentiment)

    # Load data tambahan (gojek dan tokopedia)
    gojek_df = pd.read_csv('data/gojek.csv')
    tokopedia_df = pd.read_csv('data/tokopedia.csv')

    # Tambahkan kolom category untuk gojek dan tokopedia jika diperlukan
    gojek_df['category'] = gojek_df['score'].apply(classify_sentiment)
    tokopedia_df['category'] = tokopedia_df['score'].apply(classify_sentiment)

except Exception as e:
    df = None
    gojek_df = None
    tokopedia_df = None
    print(f"Error loading CSV: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reviews')
def reviews():
    if df is not None:
        search_query = request.args.get('search', '').lower()
        filtered_df = df

        if search_query:
            filtered_df = df[
                df['userName'].str.lower().str.contains(search_query) |
                df['content'].str.lower().str.contains(search_query) |
                df['category'].str.lower().str.contains(search_query) |
                df['score'].astype(str).str.contains(search_query)
            ]

        reviews_data = filtered_df[['userName', 'score', 'content', 'category']].to_dict(orient='records')
        return render_template('reviews.html', reviews=reviews_data, search_query=search_query)
    else:
        return "Data tidak tersedia."

@app.route('/visualize')
def visualize():
    if df is not None and gojek_df is not None and tokopedia_df is not None:
        if 'category' not in df.columns:
            return "Kolom 'category' tidak ditemukan di data."

        # Hitung jumlah setiap kategori untuk semua dataset
        flip_counts = df['category'].value_counts().to_dict()
        gojek_counts = gojek_df['category'].value_counts().to_dict()
        tokopedia_counts = tokopedia_df['category'].value_counts().to_dict()

        # Gabungkan label dari semua dataset (asumsi kategori sama)
        labels = list(flip_counts.keys())
        
        # Nilai untuk masing-masing dataset
        flip_values = list(flip_counts.values())
        gojek_values = list(gojek_counts.values())
        tokopedia_values = list(tokopedia_counts.values())
        
        search_query = request.args.get('search', '').lower()
        filtered_df = df

        if search_query:
            filtered_df = df[
                df['userName'].str.lower().str.contains(search_query) |
                df['content'].str.lower().str.contains(search_query) |
                df['category'].str.lower().str.contains(search_query) |
                df['score'].astype(str).str.contains(search_query)
            ]

        reviews_data = filtered_df[['userName', 'score', 'content', 'category']].to_dict(orient='records')
        
        return render_template('visualize.html', 
                             labels=labels, 
                             flip_values=flip_values,
                             reviews=reviews_data, 
                             search_query=search_query)
    else:
        return "Data tidak tersedia."
@app.route('/visualize_gojek')
def visualizego():
    if df is not None and gojek_df is not None and tokopedia_df is not None:
        if 'category' not in df.columns:
            return "Kolom 'category' tidak ditemukan di data."

        # Hitung jumlah setiap kategori untuk semua dataset
        flip_counts = df['category'].value_counts().to_dict()
        gojek_counts = gojek_df['category'].value_counts().to_dict()
        tokopedia_counts = tokopedia_df['category'].value_counts().to_dict()

        # Gabungkan label dari semua dataset (asumsi kategori sama)
        labels = list(flip_counts.keys())
        
        # Nilai untuk masing-masing dataset
        flip_values = list(flip_counts.values())
        gojek_values = list(gojek_counts.values())
        tokopedia_values = list(tokopedia_counts.values())
        
        search_query = request.args.get('search', '').lower()
        filtered_df = df

        if search_query:
            filtered_df = df[
                df['userName'].str.lower().str.contains(search_query) |
                df['content'].str.lower().str.contains(search_query) |
                df['category'].str.lower().str.contains(search_query) |
                df['score'].astype(str).str.contains(search_query)
            ]

        reviews_data = filtered_df[['userName', 'score', 'content', 'category']].to_dict(orient='records')
        
        return render_template('visualize.html', 
                             labels=labels, 
                             gojek_values=gojek_values,
                             reviews=reviews_data, 
                             search_query=search_query)
    else:
        return "Data tidak tersedia."
@app.route('/visualize_tokopedia')
def visualizeto():
    if df is not None and gojek_df is not None and tokopedia_df is not None:
        if 'category' not in df.columns:
            return "Kolom 'category' tidak ditemukan di data."

        # Hitung jumlah setiap kategori untuk semua dataset
        flip_counts = df['category'].value_counts().to_dict()
        gojek_counts = gojek_df['category'].value_counts().to_dict()
        tokopedia_counts = tokopedia_df['category'].value_counts().to_dict()

        # Gabungkan label dari semua dataset (asumsi kategori sama)
        labels = list(flip_counts.keys())
        
        # Nilai untuk masing-masing dataset
        flip_values = list(flip_counts.values())
        gojek_values = list(gojek_counts.values())
        tokopedia_values = list(tokopedia_counts.values())
        
        search_query = request.args.get('search', '').lower()
        filtered_df = df

        if search_query:
            filtered_df = df[
                df['userName'].str.lower().str.contains(search_query) |
                df['content'].str.lower().str.contains(search_query) |
                df['category'].str.lower().str.contains(search_query) |
                df['score'].astype(str).str.contains(search_query)
            ]

        reviews_data = filtered_df[['userName', 'score', 'content', 'category']].to_dict(orient='records')
        
        return render_template('visualize.html', 
                             labels=labels, 
                             tokopedia_values=tokopedia_values,
                             reviews=reviews_data,
                             search_query=search_query)
    else:
        return "Data tidak tersedia."

if __name__ == '__main__':
    app.run(debug=True)