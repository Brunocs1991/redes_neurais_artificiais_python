entradas = [-1, 7, 5]
pesos = [0.8, 0.1, 0]


def soma(e, p):
    # print(e, p)
    s = 0
    for i in range(3):
        s += e[i] * p[i]
    return s


s = soma(entradas, pesos)
print(s)


def stepFunction(s):
    if (s >= 1):
        return 1
    return 0


r = stepFunction(s)
print(r)
