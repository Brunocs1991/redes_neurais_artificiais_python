import numpy as np


def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))


def sigmoid_derivada(sig):
    return sig * (1 - sig)


entradas = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
)

saidas = np.array(
    [
        [0],
        [1],
        [1],
        [0]
    ]
)

pesos_0 = np.array(
    [
        [-0.424, -0.740, -0.961],
        [0.358, -0.577, -0.469]
    ]
)

pesos_1 = np.array(
    [
        [-0.017],
        [-0.893],
        [0.148]
    ]
)

epocas = 100

for j in range(epocas):
    camada_entrada = entradas
    soma_sinapse_0 = np.dot(camada_entrada, pesos_0)
    camada_oculta = sigmoid(soma_sinapse_0)
    soma_sinapse_1 = np.dot(camada_oculta, pesos_1)
    camada_saida = sigmoid(soma_sinapse_1)
    erro_camada_saida = saidas - camada_saida
    media_absoluda = np.mean(np.abs(erro_camada_saida))
    derivada_saida = sigmoid_derivada(camada_saida)
    delta_saida = erro_camada_saida * derivada_saida
    print(delta_saida)
    break
