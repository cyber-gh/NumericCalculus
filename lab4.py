import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def second_derivative(expr, x):
    """
        Shortcut to calculate second derivative with sympy
    :param expr: sympy expression
    :param x: how the variable is named in the expression
    :return: the initial expression and its second derivative
    """
    ddf = expr.diff().diff()
    expr = sp.lambdify(x, expr)
    ddf = sp.lambdify(x, ddf)
    return expr, ddf


def finite_differences(X, Y):
    """
        Apply the finite differences formula with the "central" method
    :param X: values on X axis
    :param Y: values on Y axis
    :return: approximated derivative values
    """
    n = len(X)
    ans = np.zeros(n)
    ans[0] = ans[-1] = np.nan
    ans[1:-1] = [(Y[i + 1] - 2 * Y[i] + Y[i - 1]) / (X[i + 1] - X[i]) ** 2 for i in range(1, n - 1)]

    return ans


def ex1():
    eps = 1e-5
    x = sp.symbols('x')
    f = sp.cos(-0.2 * x)
    # calculate the second derivative, we'll need it to find the error
    f, s_derivative = second_derivative(f, x)
    left, right = -np.pi / 2, np.pi
    for N in range(3, 10000):
        # find the smallest N for which the error is less than eps

        # the length of the division
        h = np.abs((right - left) / N)
        # the x domain created from equally spaced intervals plus points: left - h and right + h
        x_domain = [left - h] + np.linspace(left, right, N) + [right + h]
        ys = f(x_domain)

        # apply the algorithm for given x_domain
        diff = finite_differences(x_domain, ys)
        # calculate the error using the result above and the actual second derivative
        err = max(np.abs(s_derivative(x_domain[1:-1]) - diff[1:-1]))

        if err < eps:
            print("err = ", err)
            print("N = ", N)
            plt.plot(x_domain[1: -1], s_derivative(x_domain[1: -1]), c='blue', linewidth=2, label='actual second derivative')
            plt.plot(x_domain[1: -1], diff[1: -1], c='red', linewidth=2, linestyle='-.',
                     label='progressive differences')
            plt.title('Second derivative for N = {}'.format(N))
            plt.legend()
            plt.show()

            plt.figure(1)
            plt.plot(x_domain[1:-1], np.abs(s_derivative(x_domain[1:-1]) - diff[1:-1]), label='Error function')
            plt.title("Error function")
            plt.legend()
            plt.show()
            break


def integral(f, x, method):
    """
        Calculate the integral value on the domain x of function f using the method by summing
    :param f: function to calculate integral of
    :param x: discrete domain of x values
    :param method: which method to use trapez|dreptunghi|simpsion
    :return: the value of the integral
    """
    n = x.shape[0]
    if method == "trapez":
        return sum(((x[i + 1] - x[i]) / 2) * (f(x[i]) + f(x[i + 1])) for i in range(n - 1))
    if method == "dreptunghi":
        return sum((x[i + 2] - x[i]) * f(x[i + 1]) for i in range(0, n - 2, 2))
    if method == "simpson":
        return sum(((x[i + 2] - x[i]) / 6) * (f(x[i]) + 4 * f(x[i + 1]) + f(x[i + 2])) for i in range(0, n - 2, 2))

    raise ValueError("No such method")


def ex2():
    # f is the input function
    # we calculate the integral from left to right
    # the x values will be calculated with linspace, chosen number of points is 10k
    f = lambda x: (1 / (1.4 * np.sqrt(2 * np.pi))) * np.exp(-1 * x ** 2 / (2 * 1.4 ** 2))
    left, right = -14, 14
    x = np.linspace(-left, right, 10000)
    print("Valoarea integralei folosing metoda trapez: ", integral(f, x, "trapez"))
    print("Valoarea integralei folosing metoda dreptunghi: ", integral(f, x, "dreptunghi"))
    print("Valoarea integralei folosing metoda simpson: ", integral(f, x, "simpson"))


if __name__ == '__main__':
    ex1()
    ex2()
