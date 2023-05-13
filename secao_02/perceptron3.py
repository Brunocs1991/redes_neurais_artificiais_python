import numpy as np
# Operador OR
# entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# saidas = np.array([0, 0, 0, 1])

# Operador AND
# entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# saidas = np.array([0, 1, 1, 1])

# Operador xor não linear ira ficar em loop infinito, so resolvido em perceptron multicamada
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas = np.array([0, 1, 1, 0])

pesos = np.array([0.0, 0.0])
taxa_aprendizagem = 0.1


def step_funcion(soma):
    if (soma >= 1):
        return 1
    return 0


def calcula_saida(registro):
    s = registro.dot(pesos)
    return step_funcion(s)


def treinar():
    erro_total = 1
    while (erro_total != 0):
        erro_total = 0
        for i in range(len(saidas)):
            saida_calculada = calcula_saida(np.asarray(entradas[i]))
            erro = saidas[i] - saida_calculada
            erro_total += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + \
                    (taxa_aprendizagem * entradas[i][j] * erro)
                print(f'Peso atualizado: {str(pesos[j])}')
        print(f'Total de erros: {str(erro_total)}')


treinar()
print('Rede neural treinada')
print(calcula_saida(entradas[0]))
print(calcula_saida(entradas[1]))
print(calcula_saida(entradas[2]))
print(calcula_saida(entradas[3]))
