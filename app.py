# -*- coding: utf-8 -*-
"""
@author: fred_
"""

from sklearn.preprocessing import StandardScaler
import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.neural_network import multilayer_perceptron


scaler = StandardScaler()

app = Flask(__name__)
svm = pickle.load(open('svm_model.sav', 'rb'))
random_forest = pickle.load(open('random_forest_model.sav', 'rb'))
rna_model = pickle.load(open('mlp_model.sav','rb'))

@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/indexRF')
def rf():
    return render_template('indexRF.html')

@app.route('/indexRNA')
def rna():
    return render_template('indexRNA.html')

@app.route('/predictSVM', methods=['POST','GET'])
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
        return render_template('index.html', prediction_text_svm ='Resultado: Vai pagar')
    elif resposta_svm == 1:
        return render_template('index.html', prediction_text_svm ='Resultado: Não vai pagar')
    else:
        return render_template('index.html', prediction_text_svm ='Resultado: ')

@app.route('/predictRF',methods=['POST','GET'])
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
        return render_template('indexRF.html', prediction_text_rf ='Resultado: Vai pagar')
    elif resposta_rf == 1:
        return render_template('indexRF.html', prediction_text_rf ='Resultado: Não vai pagar')
    else:
        return render_template('indexRF.html', prediction_text_rf ='Resultado: ')
    

@app.route('/predictRNA',methods=['POST','GET'])
def predict_rna():
    renda = request.form['inputRendaRNA']
    idade = request.form['inputIdadeRNA']
    credito = request.form['inputCreditoRNA']
    novo_registro = [[renda, idade, credito]]
    novo_registro = np.asarray(novo_registro)
    novo_registro = novo_registro.reshape(-1, 1)
    novo_registro = scaler.fit_transform(novo_registro)
    novo_registro = novo_registro.reshape(-1, 3)
    
    resposta_rna = rna_model.predict(novo_registro)
    if resposta_rna == 0:
        return render_template('indexRNA.html', prediction_text_rna ='Resultado: Vai pagar')
    elif resposta_rna == 1:
        return render_template('indexRNA.html', prediction_text_rna ='Resultado: Não vai pagar')
    else:
        return render_template('indexRNA.html', prediction_text_rna ='Resultado: ')
    

if __name__ == "__main__":
    app.run(debug=True)
