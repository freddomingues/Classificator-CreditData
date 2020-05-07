# -*- coding: utf-8 -*-
"""
@author: fred_
"""

from sklearn.preprocessing import StandardScaler
import numpy as np
from flask import Flask, request, render_template
import pickle

scaler = StandardScaler()

app = Flask(__name__)
svm = pickle.load(open('svm_model.sav', 'rb'))
rna = pickle.load(open('rna_model.sav', 'rb'))
random_forest = pickle.load(open('random_forest_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict_svm():
    renda = request.form['inputRendaSVM']
    idade = request.form['inputIdadeSVM']
    credito = request.form['inputCreditoSVM']
    novo_registro = [[renda, idade, credito]]
    novo_registro = np.asarray(novo_registro)
    novo_registro = novo_registro.reshape(-1, 1)
    novo_registro = scaler.fit_transform(novo_registro)
    novo_registro = novo_registro.reshape(-1, 3)
    
    resposta_svm = svm.predict(novo_registro)
    if resposta_svm == 0:
        output = ' Vai pagar - empréstimo liberado'
    elif resposta_svm == 1:
        output = ' Não vai pagar - empréstimo negado'
    
    return render_template('index.html', prediction_text_svm ='Resultado: {}'.format(output))


@app.route('/',methods=['POST'])
def predict_rf():
    renda = request.form['inputRendaRF']
    idade = request.form['inputIdadeRF']
    credito = request.form['inputCreditoRF']
    novo_registro = [[renda, idade, credito]]
    novo_registro = np.asarray(novo_registro)
    novo_registro = novo_registro.reshape(-1, 1)
    novo_registro = scaler.fit_transform(novo_registro)
    novo_registro = novo_registro.reshape(-1, 3)
    
    resposta_rf = random_forest.predict(novo_registro)
    if resposta_rf == 0:
        output = 'Vai pagar - empréstimo liberado'
    elif resposta_rf == 1:
        output = 'Não vai pagar - empréstimo negado'
    
    return render_template('index.html', prediction_text_rf ='Resultado: {}'.format(output))

@app.route('/',methods=['POST'])
def predict_rna():
    renda = request.form['inputRendaRNA']
    idade = request.form['inputIdadeRNA']
    credito = request.form['inputCreditoRNA']
    novo_registro = [[renda, idade, credito]]
    novo_registro = np.asarray(novo_registro)
    novo_registro = novo_registro.reshape(-1, 1)
    novo_registro = scaler.fit_transform(novo_registro)
    novo_registro = novo_registro.reshape(-1, 3)
    
    resposta_rna = rna.predict(novo_registro)
    if resposta_rna == 0:
        output = 'Vai pagar - empréstimo liberado'
    elif resposta_rna == 1:
        output = 'Não vai pagar - empréstimo negado'
    
    return render_template('index.html', prediction_text_rna ='Resultado: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
