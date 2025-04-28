from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load data CSV saat server start
try:
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

except Exception as e:
    df = None
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
    if df is not None:
        if 'category' not in df.columns:
            return "Kolom 'category' tidak ditemukan di data."

        # Hitung jumlah setiap kategori
        category_counts = df['category'].value_counts().to_dict()
        labels = list(category_counts.keys())
        values = list(category_counts.values())

        return render_template('visualize.html', labels=labels, values=values)
    else:
        return "Data tidak tersedia."

if __name__ == '__main__':
    app.run(debug=True)
