from flask import (
    Flask, render_template, request,
    redirect, url_for, session
)

import numpy as np
from bidict import bidict
from random import choice
from tensorflow import keras

ENCODER = bidict({
    'අ':0, 'ආ':1, 'ඇ': 2, 'ඈ':3, 'ඉ':4, 'ඊ':5, 'උ':6,
    'එ':7, 'ඒ':8, 'ඔ':9, 'ඕ':10, 'ක':11, 'ග':12, 'ච':13, 
    'ජ':14, 'ට':15, 'ඩ':16, 'ණ':17, 'ත':18, 'ද':19, 'න':20,
    'ප':21, 'බ':22, 'ම':23, 'ය':24, 'ර':25, 'ල':26, 'ව':27,
    'ස':28, 'හ':29, 'ළ':30
})

app = Flask(__name__)
app.secret_key = 'sinhala_alphabet_practice'

@app.route('/')
def index():
    session.clear()
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")    

@app.route('/help')
def help():
    return render_template("help.html")       

@app.route('/add-data', methods=['GET'])
def add_data_get():
    message = session.get('message', '')
    letter = choice(list(ENCODER.keys()))
    
    return render_template("addData.html", letter=letter, message=message)

@app.route('/add-data', methods=['POST'])
def add_data_post():
    try:
        label = request.form['letter']
        labels = np.load('data/labels.npy')
        add_label = ENCODER[label]
        labels = np.append(labels, add_label)
        np.save('data/labels.npy', labels)

        pixels = request.form['pixels']
        pixels = pixels.split(',')
        img = np.array(pixels).astype(float).reshape(1, 50, 50)
        imgs = np.load('data/images.npy')
        imgs = np.vstack([imgs, img])
        np.save('data/images.npy', imgs)

        session['message'] = f'"{label}" Added to the dataset'

        return redirect(url_for('add_data_get'))

    except Exception as e:
        print(e)
        return render_template('error.html')

@app.route('/practice', methods=['GET'])
def practice_get():
    letter = choice(list(ENCODER.keys()))
    return render_template("practice.html", letter=letter, correct='')

@app.route('/practice', methods=['POST'])
def practice_post():
    try:
        letter = request.form['letter']

        pixels = request.form['pixels']
        pixels = pixels.split(',')
        img = np.array(pixels).astype(float).reshape(1, 50, 50, 1)

        model = keras.models.load_model('model.h5')

        pred_letter = np.argmax(model.predict(img), axis=-1)
        pred_letter = ENCODER.inverse[pred_letter[0]]

        correct = 'yes' if pred_letter == letter else 'no'
        letter = choice(list(ENCODER.keys()))

        return render_template("practice.html", letter=letter, correct=correct)

    except Exception as e:
        print(e)
        return render_template('error.html')

@app.route('/guess', methods=['GET'])
def guess_get():
    return render_template("guess.html", letter='')

@app.route('/guess', methods=['POST'])
def guess_post():
    try:
        letter = request.form['letter']
        pixels = request.form['pixels']
        pixels = pixels.split(',')
        img = np.array(pixels).astype(float).reshape(1, 50, 50, 1)

        model = keras.models.load_model('model.h5')

        pred_letter = np.argmax(model.predict(img), axis=-1)
        pred_letter = ENCODER.inverse[pred_letter[0]]

        return render_template("guess.html", letter=pred_letter)

    except Exception as e:
        print(e)
        return render_template('error.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host = '0.0.0.0')