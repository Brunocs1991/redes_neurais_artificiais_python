import numpy as np
from sklearn import datasets


def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))


def sigmoid_derivada(sig):
    return sig * (1 - sig)


base = datasets.load_breast_cancer()
entradas = base.data
valores_saida = base.target
saidas = np.empty([569, 1], dtype=int)
for i in range(569):
    saidas[i] = valores_saida[i]


pesos_0 = 2*np.random.random((30, 5)) - 1
pesos_1 = 2*np.random.random((5, 1)) - 1

epocas = 10000
taxa_aprendizagem = 0.3
momento = 1

for j in range(epocas):
    camada_entrada = entradas
    soma_sinapse_0 = np.dot(camada_entrada, pesos_0)
    camada_oculta = sigmoid(soma_sinapse_0)

    soma_sinapse_1 = np.dot(camada_oculta, pesos_1)
    camada_saida = sigmoid(soma_sinapse_1)

    erro_camada_saida = saidas - camada_saida
    media_absoluda = np.mean(np.abs(erro_camada_saida))
    print(f'Erro: {media_absoluda}')

    derivada_saida = sigmoid_derivada(camada_saida)
    delta_saida = erro_camada_saida * derivada_saida

    pesos_1_transpostas = pesos_1.T
    delta_saida_x_pesos = delta_saida.dot(pesos_1_transpostas)
    delta_camada_oculta = delta_saida_x_pesos * sigmoid_derivada(camada_oculta)

    camada_oculta_transposta = camada_oculta.T
    pesos_novo_1 = camada_oculta_transposta.dot(delta_saida)
    pesos_1 = (pesos_1 * momento) + (pesos_novo_1 * taxa_aprendizagem)

    camada_entrada_transposta = camada_entrada.T
    pesos_novo_0 = camada_entrada_transposta.dot(delta_camada_oculta)
    pesos_0 = (pesos_0 * momento) + (pesos_novo_0 * taxa_aprendizagem)
