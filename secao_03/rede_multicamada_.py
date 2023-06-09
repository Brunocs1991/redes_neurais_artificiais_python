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

# pesos_0 = np.array(
#     [
#         [-0.424, -0.740, -0.961],
#         [0.358, -0.577, -0.469]
#     ]
# )

# pesos_1 = np.array(
#     [
#         [-0.017],
#         [-0.893],
#         [0.148]
#     ]
# )

pesos_0 = 2*np.random.random((2, 3)) - 1
pesos_1 = 2*np.random.random((3, 1)) - 1
epocas = 10000
taxa_aprendizagem = 0.6
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
