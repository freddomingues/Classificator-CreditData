# Classificator CreditData - Sistema de Classificação de Crédito com Machine Learning

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Arquitetura do Sistema](#arquitetura-do-sistema)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Uso](#uso)
- [Modelos de Machine Learning](#modelos-de-machine-learning)
- [API e Endpoints](#api-e-endpoints)
- [Estrutura de Dados](#estrutura-de-dados)
- [Deploy](#deploy)
- [Desenvolvimento](#desenvolvimento)
- [Testes e Validação](#testes-e-validação)

---

## 🎯 Visão Geral

Sistema de classificação de crédito baseado em Machine Learning que prevê se um cliente de banco irá pagar ou não um empréstimo solicitado. A aplicação utiliza três algoritmos de ML (SVM, Random Forest e Redes Neurais Artificiais) que foram selecionados após análise comparativa de desempenho com validação cruzada.

### Objetivo

Classificar novos clientes em duas categorias:
- **Classe 0**: Cliente vai pagar o empréstimo
- **Classe 1**: Cliente não vai pagar o empréstimo

### Dataset

O sistema foi treinado com um dataset contendo **2.000 registros** de dados de crédito (`credit-data.csv`), contendo as seguintes características:
- Renda anual
- Idade
- Valor do empréstimo
- Classe (variável alvo)

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────┐
│   Frontend      │
│  (Bootstrap)    │
└────────┬────────┘
         │ HTTP POST
         ▼
┌─────────────────┐
│   Flask App     │
│   (Backend)     │
└────────┬────────┘
         │
         ├──► StandardScaler (Normalização)
         │
         ├──► SVM Model
         ├──► Random Forest Model
         └──► MLP Model (Redes Neurais)
```

### Fluxo de Processamento

1. **Entrada**: Usuário fornece dados (renda, idade, valor do empréstimo)
2. **Pré-processamento**: Dados são normalizados usando `StandardScaler`
3. **Predição**: Modelo ML selecionado faz a classificação
4. **Saída**: Resultado retornado ao usuário via interface web

---

## 🛠️ Tecnologias Utilizadas

### Backend
- **Python 3.x**: Linguagem principal
- **Flask 1.1.1**: Framework web para API e interface
- **Gunicorn 19.9.0**: Servidor WSGI para produção

### Machine Learning
- **scikit-learn ≥0.18**: Biblioteca de ML
  - `SVC`: Support Vector Machine
  - `RandomForestClassifier`: Random Forest
  - `MLPClassifier`: Multi-Layer Perceptron (Redes Neurais)
- **NumPy ≥1.9.2**: Operações numéricas
- **Pandas ≥0.19**: Manipulação de dados
- **scipy ≥0.15.1**: Computação científica

### Pré-processamento
- **StandardScaler**: Normalização de features
- **SimpleImputer**: Tratamento de valores faltantes

### Frontend
- **Bootstrap 4.4.1**: Framework CSS para interface responsiva
- **HTML5/CSS3**: Estrutura e estilização

### Ferramentas
- **Pickle**: Serialização de modelos ML
- **Heroku**: Plataforma de deploy

---

## 📁 Estrutura do Projeto

```
Classificator-CreditData/
│
├── app.py                      # Aplicação Flask principal
├── models.py                   # Treinamento e criação dos modelos ML
├── classificator.py           # Script de teste de classificação
├── requirements.txt           # Dependências do projeto
├── Procfile                   # Configuração para deploy no Heroku
│
├── credit-data.csv            # Dataset de treinamento (2k registros)
├── Teste_estatistico.xlsx    # Resultados dos testes estatísticos
│
├── svm_model.sav              # Modelo SVM serializado
├── random_forest_model.sav    # Modelo Random Forest serializado
├── mlp_model.sav              # Modelo MLP (Redes Neurais) serializado
│
├── templates/                 # Templates HTML
│   ├── index.html             # Interface para SVM
│   ├── indexRF.html           # Interface para Random Forest
│   ├── indexRNA.html          # Interface para Redes Neurais
│   └── debug.log
│
├── screenshots/               # Imagens de documentação
│   ├── APP.jpeg
│   ├── desempenho_algoritmos.jpeg
│   └── testes_estatisticos.jpeg
│
└── README.md                  # Este arquivo
```

---

## 📦 Pré-requisitos

- **Python 3.6+**
- **pip** (gerenciador de pacotes Python)
- **Git** (opcional, para versionamento)

---

## 🔧 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/freddomingues/Classificator-CreditData.git
cd Classificator-CreditData
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuração

### Modelos Pré-treinados

O projeto já inclui três modelos pré-treinados:
- `svm_model.sav`
- `random_forest_model.sav`
- `mlp_model.sav`

Estes modelos são carregados automaticamente quando a aplicação inicia.

### Treinar Novos Modelos (Opcional)

Se desejar retreinar os modelos com novos dados:

```bash
python models.py
```

Este script irá:
1. Carregar o dataset `credit-data.csv`
2. Tratar valores faltantes e inconsistências
3. Normalizar os dados
4. Treinar os três modelos
5. Salvar os modelos serializados usando Pickle

**Parâmetros dos Modelos:**

- **SVM**: `kernel='rbf'`, `C=2.0`
- **Random Forest**: `n_estimators=40`, `criterion='entropy'`
- **MLP**: 
  - `hidden_layer_sizes=(100)`
  - `activation='relu'`
  - `solver='adam'`
  - `max_iter=1000`
  - `learning_rate_init=0.001`
  - `batch_size=200`

---

## 🚀 Uso

### Executar Localmente

```bash
python app.py
```

A aplicação estará disponível em: `http://localhost:5000`

### Acessar as Interfaces

- **SVM**: `http://localhost:5000/index`
- **Random Forest**: `http://localhost:5000/indexRF`
- **Redes Neurais**: `http://localhost:5000/indexRNA`

### Fazer uma Predição

1. Acesse uma das interfaces acima
2. Preencha os campos:
   - **Renda Anual**: Valor numérico (ex: 50000)
   - **Idade**: Valor numérico (ex: 40)
   - **Valor do Empréstimo**: Valor numérico (ex: 5000)
3. Clique em "Prever"
4. O resultado será exibido na mesma página

### Testar via Script

```bash
python classificator.py
```

Este script testa os três modelos com um registro de exemplo: `[50000, 40, 5000]`

---

## 🤖 Modelos de Machine Learning

### Algoritmos Testados

Durante o desenvolvimento, foram testados 7 algoritmos diferentes:

1. **Naive Bayes** (GaussianNB)
2. **Árvore de Decisão** (DecisionTreeClassifier)
3. **Regressão Logística** (LogisticRegression)
4. **SVM** (Support Vector Machine)
5. **KNN** (K-Nearest Neighbors)
6. **Random Forest** (RandomForestClassifier)
7. **Redes Neurais Artificiais** (MLPClassifier)

### Seleção dos Melhores Modelos

Os três melhores algoritmos foram selecionados através de:

- **Validação Cruzada**: Stratified K-Fold com 10 folds
- **Repetições**: 30 execuções com diferentes sementes aleatórias
- **Métrica**: Accuracy Score (Precisão)
- **Testes Estatísticos**: Comparação de desempenho entre algoritmos

**Resultado**: Os três algoritmos com melhor desempenho foram:
1. **Redes Neurais Artificiais (MLP)**
2. **Random Forest**
3. **SVM**

### Métricas de Desempenho

Os resultados detalhados podem ser consultados em:
- `Teste_estatistico.xlsx`
- Screenshot: `screenshots/desempenho_algoritmos.jpeg`
- Screenshot: `screenshots/testes_estatisticos.jpeg`

---

## 🔌 API e Endpoints

### Endpoints Disponíveis

#### GET `/index`
Retorna a interface web para classificação com SVM.

**Resposta**: Template HTML (`index.html`)

---

#### GET `/indexRF`
Retorna a interface web para classificação com Random Forest.

**Resposta**: Template HTML (`indexRF.html`)

---

#### GET `/indexRNA`
Retorna a interface web para classificação com Redes Neurais.

**Resposta**: Template HTML (`indexRNA.html`)

---

#### POST `/predictSVM`
Faz predição usando o modelo SVM.

**Parâmetros (form-data):**
- `inputRendaSVM` (float): Renda anual do cliente
- `inputIdadeSVM` (int): Idade do cliente
- `inputCreditoSVM` (float): Valor do empréstimo solicitado

**Resposta**: Template HTML com resultado da predição
- `0`: "Vai pagar"
- `1`: "Não vai pagar"

**Exemplo de uso:**
```bash
curl -X POST http://localhost:5000/predictSVM \
  -d "inputRendaSVM=50000&inputIdadeSVM=40&inputCreditoSVM=5000"
```

---

#### POST `/predictRF`
Faz predição usando o modelo Random Forest.

**Parâmetros (form-data):**
- `inputRendaRF` (float): Renda anual do cliente
- `inputIdadeRF` (int): Idade do cliente
- `inputCreditoRF` (float): Valor do empréstimo solicitado

**Resposta**: Template HTML com resultado da predição

---

#### POST `/predictRNA`
Faz predição usando o modelo de Redes Neurais.

**Parâmetros (form-data):**
- `inputRendaRNA` (float): Renda anual do cliente
- `inputIdadeRNA` (int): Idade do cliente
- `inputCreditoRNA` (float): Valor do empréstimo solicitado

**Resposta**: Template HTML com resultado da predição

---

## 📊 Estrutura de Dados

### Dataset de Entrada (`credit-data.csv`)

O dataset contém as seguintes colunas:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `clientid` | int | ID único do cliente |
| `income` | float | Renda anual do cliente |
| `age` | int | Idade do cliente |
| `loan` | float | Valor do empréstimo solicitado |
| `default` | int | Classe (0 = paga, 1 = não paga) |

### Pré-processamento

1. **Tratamento de Idade Negativa**: Valores de idade negativos são substituídos pela média (40.92)
2. **Tratamento de Valores Faltantes**: Utiliza `SimpleImputer` com estratégia de média
3. **Normalização**: Aplica `StandardScaler` para padronizar as features

### Formato de Entrada para Predição

```python
novo_registro = [[renda_anual, idade, valor_emprestimo]]
# Exemplo: [[50000, 40, 5000]]
```

### Formato de Saída

```python
0  # Cliente vai pagar
1  # Cliente não vai pagar
```

---

## 🌐 Deploy

### Deploy no Heroku

O projeto está configurado para deploy no Heroku.

#### 1. Instalar Heroku CLI

```bash
# macOS
brew install heroku/brew/heroku

# Ou baixe em: https://devcenter.heroku.com/articles/heroku-cli
```

#### 2. Login no Heroku

```bash
heroku login
```

#### 3. Criar Aplicação

```bash
heroku create nome-da-aplicacao
```

#### 4. Deploy

```bash
git push heroku main
```

#### 5. Configuração

O `Procfile` já está configurado:
```
web: gunicorn app:app
```

### Aplicação em Produção

Aplicação disponível em: https://credit-predict.herokuapp.com/index

---

## 💻 Desenvolvimento

### Estrutura do Código

#### `app.py`
- Inicializa a aplicação Flask
- Carrega os modelos ML serializados
- Define rotas para interface e predição
- Aplica normalização antes de predições

#### `models.py`
- Carrega e processa o dataset
- Treina os três modelos de ML
- Serializa os modelos usando Pickle

#### `classificator.py`
- Script de teste para validar predições
- Testa os três modelos com dados de exemplo

### Padrões de Código

- **Encoding**: UTF-8
- **Formatação**: PEP 8 (recomendado)
- **Comentários**: Docstrings em português

### Melhorias Futuras

- [ ] Adicionar validação de entrada de dados
- [ ] Implementar tratamento de erros mais robusto
- [ ] Adicionar logging estruturado
- [ ] Criar API REST com JSON
- [ ] Adicionar testes unitários
- [ ] Implementar cache para modelos
- [ ] Adicionar métricas de performance em tempo real
- [ ] Suporte a múltiplos modelos simultâneos (ensemble)

---

## 🧪 Testes e Validação

### Validação dos Modelos

Os modelos foram validados usando:

- **Método**: Stratified K-Fold Cross-Validation
- **Folds**: 10
- **Repetições**: 30 (com diferentes seeds)
- **Métrica Principal**: Accuracy Score

### Teste Manual

Para testar manualmente os modelos:

```python
python classificator.py
```

Este script testa um registro de exemplo e imprime as predições dos três modelos.

### Teste via Interface Web

1. Acesse `http://localhost:5000/index`
2. Preencha os campos com dados de teste
3. Verifique o resultado retornado

---

## 📝 Licença

Este projeto é de código aberto e está disponível para uso educacional e de pesquisa.

---

## 👤 Autor

**Frederico A. Domingues**

- GitHub: [@freddomingues](https://github.com/freddomingues)
- Website: [https://freddomingues.github.io](https://freddomingues.github.io)

---

## 📚 Referências

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Heroku Python Support](https://devcenter.heroku.com/articles/python-support)

---

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

---

**Última atualização**: 2024
