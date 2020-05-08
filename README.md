# ML para prever empréstimos
Aplicação de Machine Learning para prever se um novo cliente de um banco vai pagar o empréstimo solicitado ou não.
Para criar os modelos de ML foram analisados os dados do dataset `credit-data.csv` com 2k registros. Para o pré-processamento dos dados foram usadas as bibliotecas:
  -	Pandas
  -	NumPy
  -	Scikit-learn
  
Para que a aplicação faça a previsão, é necessário informar os dados de entrada, sendo eles:
  -	Renda anual
  -	Idade
  -	Valor do empréstimo
  
Foram utilizados diversos algoritmos de classificação para encontrar os melhores, sendo eles:
  -	Naive Bayes
  -	Árvore de Decisão
  -	Regressão Logística
  -	SVM
  -	KnN
  -	Random Forest
  -	Redes Neurais
  
A média do desempenho de cada pode ser observado no arquivo `teste_estatístico.csv` e/ou no screenshot apresentado abaixo:

![desempenho_algoritmos](https://user-images.githubusercontent.com/44576048/81318103-1a43c600-9064-11ea-9dcd-865cf05631b4.jpeg)

Desses, os três algoritmos com melhores desempenho de acordo com o método de validação cruzada com 30 repetições de sementes geradoras e testes estatísticos, foram: 
  -	Redes Neurais Artificiais
  -	Random Forest
  -	SVM

![testes_estatisticos](https://user-images.githubusercontent.com/44576048/81318160-2f205980-9064-11ea-9133-e6e285b51c5a.jpeg)

Com isso, foram criados os modelos de ML desses 3 melhores algoritmos.
Para a criação da aplicação web para previsão de novos registros, foram utilizadas as seguintes ferramentas:
  -	Biblioteca Pickle para armazenar/ler os modelos no/do disco;
  -	Flask para fazer o back-end da aplicação;
  -	Bootstrap para o front-end da aplicação;
  -	GitHub para hospedagem do código;
  -	Heroku para fazer o deploy da aplicação.

A aplicação web faz a classificação do novo registro utilizando os 3 melhores modelos obtidos. 

![APP](https://user-images.githubusercontent.com/44576048/81318184-3a738500-9064-11ea-9672-535f78a92c8a.jpeg)

Disponível em: https://credit-predict.herokuapp.com/index

