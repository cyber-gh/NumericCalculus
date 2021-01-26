# Tema 1 Soltan Gheorghe 334

import numpy as np
import matplotlib.pyplot as plot


def newton_raphson_method(f, derivative, x0, eps):
    while True:
        x1 = x0 - f(x0) / derivative(x0)
        if abs(x1 - x0) / abs(x0) < eps:
            break

        x0 = x1

    sol = float("{:.8f}".format(x1))
    return sol


def ex1():
    """
    Conditie: Sa se gaseasca o aprixamre a valorii sqrt(5) cu precizie de 7 zecimalew
    Fie ecuatia sqrt(5) = x <=> x^2 - 5 = 0
    Avem function f(x) = x^2 - 5 = 0
    Derivata f'(x) = 2*x
    Alegem un interval [a, b] = [2, 3]
    Functia este continua si derivabila pe acest interval, derivata nu se anuleaza
    f(a) * f(b) = f(2) * f(3) = -1 * 4 = -4 < 0
    Alegem x0 = 2.5


    :return: valoarea pentru radical din 5
    """
    x0 = 2.5
    eps = 1e-7
    f = lambda x: x ** 2 - 5
    derivative = lambda x: 2 * x
    return newton_raphson_method(f, derivative, x0, eps)


def bisection_method(a, b, f, eps):
    a0 = a
    b0 = b
    x0 = (a0 + b0) / 2
    iterations = int(np.log2((b - a) / eps) - 1) + 1
    x1, a1, b1 = x0, a0, b0

    for _ in range(0, iterations):
        if f(x0) == 0:
            break
        elif f(a0) * f(x0) < 0:
            a1 = a0
            b1 = x0
            x1 = (a1 + b1) / 2
        elif f(a0) * f(x0) > 0:
            a1 = x0
            b1 = b0
            x1 = (a1 + b1) / 2

        x0, a0, b0 = x1, a1, b1

    return x0


def ex2():
    """
    Aplicam metoda bisectiei
    Avem functia f(x) = e^(x-2) - cos(e^(x-2) - 1
    f1(x) = e^(x-2)
    f2(x) = cos(e^(x-2) + 1
    f este o functie continua, evident (e^x functie continua, cos(x) functie continua, compunerea a 2 functii continue e functie continua)
    f'(x) = e^(x-2) * (1 + sin(e^(x-2))
    -1 <= sin(x) <= 1 si e^x > 0 <=> f'(x) >= 0 orice x din R <=>
    f este strict crescatore <=>
    f are solutie unica(daca are solutie)

    Alegem un interval [a, b] = [0, 4],
    f(0) * f(4) = -11.023.... < 0
    Deci se respecta conditia f(a) * f(b) < 0

    :return: solutia ecuatiei
    """
    f = lambda x: np.exp(x - 2) - np.cos(np.exp(x - 2)) - 1
    f1 = lambda x: np.exp(x - 2)
    f2 = lambda x: np.cos(np.exp(x - 2)) + 1
    sol = bisection_method(0, 4, f, 1e-6)

    interval = np.linspace(0, 4, 100)
    y1 = f1(interval)
    y2 = f2(interval)
    plot.plot(interval, y1, 'r', label='f1(x) = e^(x-2)')
    plot.plot(interval, y2, 'g', label='f2(x) = cos(e^(x-2) + 1')
    plot.legend(loc="upper left")
    plot.plot(sol, f1(sol), 'o')
    plot.show()
    return sol


def false_position_method(f, a, b, eps):
    x = (a * f(b) - b * f(a)) / (f(b) - f(a))
    x_prev = x
    iterations = 0
    while True:
        iterations += 1
        if f(x_prev) == 0:
            x = x_prev
            break
        elif f(a) * f(x_prev) < 0:
            a, b = a, x
            x = (a * f(b) - b * f(a)) / (f(b) - f(a))
        elif f(a) * f(x_prev) > 0:
            a, b = x, b
            x = (a * f(b) - b * f(a)) / (f(b) - f(a))

        if abs(x - x_prev) < abs(x_prev) * eps:
            break
        x_prev = x

    return x, iterations


def ex3():
    """
    Avem functia f(x)  = x^3 + 8 * x^2 + 15 * x pe interval [-5, 5], eps = 1e-5
    Pentru a aplica metoda pozitiei false, va trebui sa gasim un interval [a,b] inclus in [-5,5], cu proprietatea
        ca f(a) * f(b) < 0, iar nici f', nici f'' nu se anuleaza pe intervalul [a, b]

    f'(x) = 3 * x ^ 2 + 16 * x  + 15 ( se anuleaza in -4.11, -1.2_
    f''(x) = 6 * x + 16 (se anuleaza in -2.(6))
    Dupa ce am facut graficul functiei, am ales intervalele [-6, -4.5], [-4, -2.7], [-1, 1]
    care verifica toate cele 3 condtitii ale teoremei, deci puteam aplica metoda pozitiei false
    :return: solutiile functiei
    """
    f = lambda x: x ** 3 + 8 * x ** 2 + 15 * x
    # interval = np.linspace(-5, 5, 100)
    # y = f(interval)
    # plot.plot(interval, y, 'r', label='f(x)  = x^3 + 8 * x^2 + 15 * x')
    # plot.legend(loc="upper left")
    # plot.show()
    eps = 1e-5
    (x1, n1), (x2, n2), (x3, n3) = false_position_method(f, -6, -4.5, eps), false_position_method(f, -4, -2.7,
                                                                                                  eps), false_position_method(
        f, -1, 1, eps)

    print("Solutia x1 = {:.5f} in {} iteratii".format(x1, n1))
    print("Solutia x2 = {:.5f} in {} iteratii".format(x2, n2))
    print("Solutia x3 = {:.5f} in {} iteratii".format(x3, n3))


def secant_method(f, a, b, x0, x1, eps):
    iterations = 0
    x2 = 0

    while abs(x1 - x0) >= abs(x0) * eps:
        iterations += 1
        if f(x1) - f(x0) == 0:
            break
        x2 = (x0 * f(x1) - x1 * f(x0)) / (f(x1) - f(x0))
        if x2 < a or x2 > b:
            raise ValueError("Invalid values for x0 and x1")

        x0, x1 = x1, x2

    return x2, iterations


def ex4():
    """
    Avem functia f(x) = x ^ 3 + x ^ 2 - 2 * x pe intervalul [-3, 3]
    Pentru a aplica metoda secantei, va trebui sa gasim un interval [a,b] inclus in [-3,3], cu proprietatea
        ca f(a)*f(b) < 0, iar f' nu se anuleaza pe [a, b]

    f'(x) = 3 * x ^ 2 + 2 * x - 2 (se anuleaza in -1.21, 0,54)

    Dupa ce am construit graficul functiei, am ales urmatoarele intervale
    [-3, -1.5], [-0.1, 0.1], [0.5, 1.5]]
     se observa ca verifica toate cele 2 conditii ale teoremei, deci se poate aplica functia.
    Putem alege in fiecare situatie x0 = a si x1 = b, valori care respecta conditiile teoremei
    (x0, x1 apartin [a, b])
    :return:
    """

    f = lambda x: x ** 3 + x ** 2 - 2 * x
    # interval = np.linspace(-3, 3, 100)
    # y = f(interval)
    # plot.plot(interval, y, 'r', label='f(x) = x ^ 3 + x ^ 2 - 2 * x')
    # plot.legend(loc="upper left")
    # plot.show()

    intervals = [[-3, -1.5], [-0.1, 0.1], [0.5, 1.5]]
    eps = 1e-5
    solutions = list(
        map(lambda interval: secant_method(f, interval[0], interval[1], interval[0], interval[1], eps), intervals))

    for sol, n in solutions:
        print("Solutia {} in {} iteratii".format(sol, n))


if __name__ == '__main__':
    print("Ex 1")
    print("Sol = ", ex1())

    print("Ex 2")
    print("Sol = ", ex2())

    print("Ex 3")
    ex3()

    print("Ex 4")
    ex4()
