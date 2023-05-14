from sklearn.neural_network import MLPClassifier
from sklearn import datasets

iris = datasets.load_iris()
entradas = iris.data
saidas = iris.target

rede_neural = MLPClassifier(verbose=True,
                            max_iter=5000,
                            tol=0.00001,
                            activation='logistic',
                            learning_rate_init=0.001)
rede_neural.fit(entradas, saidas)
resultado = rede_neural.predict(
    [
        [5, 7.2, 5.1, 2.2]
    ]
)

print(f"Resultado: {resultado}")
