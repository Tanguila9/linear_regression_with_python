import pandas as pd
import json

#Atualiza os pesos com base nos dados
def gradient_descent(mnow, bnow, points, Learning_rate):

    m_gradient = 0

    b_gradient = 0

    n = len(points)

    for i in range(n):

        x = points.iloc[i].horas_de_estudo

        y = points.iloc[i].resultados_da_prova

        m_gradient += - (2/n) * x * (y - (mnow * x + bnow))
        
        b_gradient += - (2/n) * (y - (mnow * x + bnow))

    m = mnow - Learning_rate * m_gradient
    
    b = bnow - Learning_rate * b_gradient

    return m,b

#dataset
datas = pd.read_csv('numbers.csv')

m = 0

b = 0

epochs = 1000
lr = 0.001

#Treinamento
for i in range(epochs):

    if i % 50 == 0:

        print(f"Epoch: {i}")

    m,b = gradient_descent(m, b, datas, lr)
    params = {
        'm':m,
        'b':b
    }
    #cria um arquivo para guardar os parâmetros do modelo
    with open('parameters.json', 'w') as f_o:
        json.dump(params, f_o)

#Função para usar o modelo
def linearegression_model(x):
    with open('parameters.json', 'r') as f_o:
        params = json.load(f_o)

    m = params['m']
    b = params['b']

    y = m * x + b

    return int(y)

print(linearegression_model(13))