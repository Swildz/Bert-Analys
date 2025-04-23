from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('dashboard.html', title='Halaman Utama')

@app.route('/about')
def about():
    return '<h1>Tentang Kami</h1><p>Ini adalah halaman tentang kami.</p>'

if __name__ == '__main__':
    app.run(debug=True)