from flask import Flask, request, render_template, redirect, url_for
import joblib
import pandas as pd
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tải mô hình đã lưu
model = joblib.load('model/spam_classifier_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message', '')
    file = request.files.get('file', None)

    if file and file.filename != '':
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        with open(filepath, 'r') as f:
            message = f.read()
        os.remove(filepath)

    if not message:
        return redirect(url_for('home'))

    data = [message]
    prediction = model.predict(data)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    return render_template('index.html', prediction=result, message=message)

@app.route('/send_contact', methods=['POST'])
def send_contact():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']
    # Ở đây, bạn có thể xử lý và lưu trữ thông tin liên hệ hoặc gửi email
    # Ví dụ: Lưu vào cơ sở dữ liệu hoặc gửi email cho quản trị viên
    return render_template('contact.html', success=True)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
