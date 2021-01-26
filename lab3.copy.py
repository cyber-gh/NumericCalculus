import numpy as np
from numpy.linalg import norm
from numpy import dot
from numpy import transpose as t
import matplotlib.pyplot as plt
from math import prod


def eval_f(A, b, x):
    """

    :param A: function component
    :param b:  function component
    :param x: value to evaluate
    :return: f(x) where f is the function represented by A & b
    """
    return 0.5 * dot(dot(x, A), t(x)) - dot(x, t(b))


def sample_grid(A, b, ranges, size=200):
    """
    Build a grid and evaluates the function in that grid
    :param A:
    :param b:
    :param ranges: range on x axis and y axis
    :param size: how many points to sample
    :return: X, Y grid and the value of f in each point of the grid
    """

    x1 = np.linspace(ranges[0][0], ranges[0][1], size)
    x2 = np.linspace(ranges[1][0], ranges[1][1], size)
    X, Y = np.meshgrid(x1, x2)
    FXY = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = np.array([
                X[i, j],
                Y[i, j]
            ])
            FXY[i, j] = eval_f(A, b, x)

    return X, Y, FXY


def draw_curves(A, b, s1, s2, nr=25):
    """

    :param A: --
    :param b: --
    :param s1: solutions generated by the gradient_descent algorithm
    :param s2: solutions generated by the conjugate gradient algorithm
    :param nr: number of levels to draw the countour
    :return:
    """
    solx = s1[-1][0]
    soly = s1[-1][1]
    diff = 0.5
    x, y, fxy = sample_grid(A, b, [[solx - diff, solx + diff], [soly - diff, soly + diff]])
    plt.plot()
    for c, n in zip(s1[:-1], s1[1:]):
        plt.plot([c[0], n[0]], [c[1], n[1]], linestyle='--', marker='o', color='r')
    for c, n in zip(s2[:-1], s2[1:]):
        plt.plot([c[0], n[0]], [c[1], n[1]], linestyle='--', marker='o', color='g')

    plt.contour(x, y, fxy, levels=nr)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.title("Gradient descent curves at each step")
    plt.show()


def gradient_descent(A, b, x0):
    """
    Applies gradient descent to find minimum of function

    :param A: matrix 2x2
    :param b: matrix 1x2
    :param x0: starting point (eg [0,0])
    :return: all the solutions, the last one being the most accurate
    """
    eps = 1e-10
    x = x0
    rk = b - dot(A, x)
    solutions = []
    while norm(rk) > eps:
        solutions.append(x)
        rk = b - dot(A, x)
        rate = dot(t(rk), rk) / dot(dot(rk, A), rk)
        x = x + rate * rk
    solutions.append(x)
    return solutions


def conjugate_gradient(A, b, x0):
    """
    Applies conjugate gradient algorithm to find the minimum of function
    :param A: matrix 2x2
    :param b: matrix 1x2
    :param x0: starting point (eg [0,0])
    :return:  all the solutions, the last one being the most accurate
    """
    eps = 1e-10
    x = x0
    dk = rk = b - dot(A, x)
    solutions = []
    while norm(rk) > eps:
        rate = dot(t(rk), rk) / dot(dot(t(dk), A), dk)
        x = x + rate * dk
        rk_n = rk - rate * dot(A, dk)
        bk = dot(t(rk_n), rk_n) / dot(t(rk), rk)
        rk = rk_n
        dk = rk + bk * dk
        solutions.append(x)

    return solutions


def determinant(matrix):
    """
    Finds determinant of matrix recursively
    :param matrix:
    :return:
    """
    m, n = matrix.shape
    if m != n:
        raise ValueError("Not a square matrix")

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    sm = 0
    for i in range(0, n):
        el = matrix[0][i]
        submatrix = np.delete(matrix, i, 1)
        submatrix = np.delete(submatrix, 0, 0)
        sm += (-1) ** (i) * el * determinant(submatrix)

    return sm


def ex1():
    A = np.asarray([[1, -5], [-5, 106]])
    b = np.asarray([3, -1])

    n = 2
    # this checks if the matrix is positive definite
    if not all(determinant(A[:idx, :idx]) > 0 for idx in range(1, n)):
        raise ValueError("Matrix is not cholesky decomposable, determinants <= 0")

    # we know that the A matrix is positive definite and symmetric, thus there is a minimum and only one

    # choosing a starting point
    x0 = np.asarray([4, 2], dtype=np.float32)
    s1 = gradient_descent(A, b, x0)
    s2 = conjugate_gradient(A, b, x0)
    print(s1)
    print(s2)

    # the diagram is not very clear as the gradient descent finds the solution too fast
    draw_curves(A, b, s1, s2)


def newton_method(x, y, point):
    """

    :param x: domain values
    :param y: value of f in all the domain values
    :param point: the point to evaluate the lagrange polynomial value for
    :return: the value/values of the lagrange polynomial in (point/points)
    """
    n = x.shape[0]
    q = np.zeros((n, n))
    for i in range(n):
        q[i][0] = y[i]
    for i in range(1, n):
        for j in range(1, i + 1):
            q[i][j] = (q[i][j - 1] - q[i - 1, j - 1]) / (x[i] - x[i - j])
    return q[0][0] + sum(q[k][k] * prod(point - x[j] for j in range(k)) for k in range(1, n))


def equal_domain(left, right, n):
    """
    default equally spaced domain
    :param left:
    :param right:
    :param n:
    :return:
    """
    return np.linspace(left, right, n)


def chebyshev_nodes_domain(left, right, n):
    """
    Chebyshev nodes are specific real algebraic numbers, namely the roots of the Chebyshev polynomials of the first kind

    I need this method because the above method of splitting equally the intervals doesn't result in a error <= 1e-5
    :param left:
    :param right:
    :param n:
    :return:
    """
    x = np.zeros(n + 1)
    x[0] = left
    x[1] = right
    for i in range(1, n):
        x[i] = 0.5 * (left + right) + 0.5 * (right - left) * np.cos(((2 * i - 1) / (2 * n - 2)) * np.pi)
    return x


def find_interpolation_nr(f, left, right, n=100, eps=1e-5, domain_generator=equal_domain):
    """
    Given the function, its domain [left, right], the error and domain_generator finds the smallest N for which the error <= eps
    :param f: function
    :param left:
    :param right:
    :param n: need a discrete number to evaluate the function for on the interval
    :param eps:
    :param domain_generator: function which generates the interpolation intervals
    :return: smallest N that satisfies the condition
    """
    domain = equal_domain(left, right, n)

    N = 2
    while True:
        xs = domain_generator(left, right, N + 1)
        ys = f(xs)

        # err = max(np.abs(f(x) - newton_method(xs, ys, x)) for x in domain)
        err = max(np.abs(f(domain) - newton_method(xs, ys, domain)))
        if err < eps:
            return N
        print(N, err)
        N += 1


def draw_polynomial(f, left, right, N=7, n=100, domain_generator=equal_domain):
    """
    For a precalculated N (minimal number of interpolation points for which the error <= eps),
    plots the original function, the polynomial function and the interpolation points
    On the next plot, there is the error function defined as |f(x) - Pn(x)| for x in domain
    :param f: function
    :param left: left margin of interval
    :param right: right margin of interval
    :param N: optimal number of interpolated points
    :param n: need a discrete number to evaluate the function for on the interval
    :param domain_generator: function which generates the interpolation intervals
    :return:
    """
    domain = equal_domain(left, right, n)
    values = f(domain)

    xs = domain_generator(left, right, N + 1)
    ys = f(xs)

    # interpolated = [newton_method(xs, ys, x) for x in domain]
    interpolated = newton_method(xs, ys, domain)

    plt.figure(0)
    plt.plot(domain, values, c="k", linewidth=2, label='Original function')
    plt.xlabel("x")
    plt.ylabel("y = f(x)")
    plt.grid()
    plt.scatter(xs, ys, marker="*", c='red', s=200, label="data")
    plt.plot(domain, interpolated, c='b', linewidth=1, linestyle="-.", label='polynomial')

    plt.legend()
    plt.show()

    plt.figure(0)
    err = [np.abs(f(x) - newton_method(xs, ys, x)) for x in domain]
    plt.grid()
    plt.plot(domain, err, c='g', linewidth=0.5, label="error function")
    plt.show()


def ex2():
    f = lambda x: 2 * np.sin(-4 * x) + 4 * np.cos(6 * x) + 4.74 * x
    # first we find the optimal N
    N = find_interpolation_nr(f, -np.pi, np.pi, domain_generator=chebyshev_nodes_domain)
    print("N = ", N)
    # then we plot the result
    draw_polynomial(f, -np.pi, np.pi, N, domain_generator=chebyshev_nodes_domain)


def nonzero_pivot_chooser(matrix):
    first_column = matrix[:, 0]
    return (first_column != 0).argmax(axis=0), 0


def first_element_chooser(matrix):
    return 0, 0


def gauss_elimination(matrix, pivot_chooser=nonzero_pivot_chooser, down=True):
    n = matrix.shape[0]
    row_swaps = []
    column_swaps = []
    for idx in range(0, n):
        # use the custom pivot chooser to get a value
        i, j = pivot_chooser(matrix[idx:, idx:])

        # add the offsets
        i += idx
        j += idx
        pivot = matrix[i][j]
        if pivot == 0:
            raise ValueError("Pivotul e 0")

        row_swaps.append((idx, i))
        column_swaps.append((idx, j))

        # swap the rows and columns if needed
        matrix[:, [idx, j]] = matrix[:, [j, idx]]
        matrix[[idx, i]] = matrix[[i, idx]]

        # determinate the direction
        way = range(idx + 1, n) if down else range(0, idx)
        for nxt in way:
            # print("nxt = ",matrix[nxt][idx])
            if matrix[nxt][idx] == 0:
                continue
            inverse = -matrix[nxt][idx] / pivot
            matrix[nxt] = np.add(inverse * matrix[idx], matrix[nxt])

    return matrix, row_swaps, column_swaps


def simplify_diagonal(matrix):
    n = matrix.shape[0]
    for idx in range(0, n):
        el = matrix[idx][idx]
        matrix[idx] = (1 / el) * matrix[idx]
        if matrix[idx][idx] < 0:
            matrix[idx] = (-1) * matrix[idx]

    return matrix


def solve_gauss(a, b, pivot_chooser=nonzero_pivot_chooser):
    converted, row_swaps, column_swaps = gauss_elimination(np.c_[a, b], pivot_chooser)
    converted, _, _ = gauss_elimination(converted, first_element_chooser, down=False)
    converted = simplify_diagonal(converted)
    solutions = converted[:, -1]

    # make the appropriate swaps on the solution if we swapped columns while running the gauss elimination
    for i, j in column_swaps:
        solutions[i], solutions[j] = solutions[j], solutions[i]

    return solutions


def spline_cubic_aproximation(x, y, left_d, right_d, points):
    n = x.shape[0]
    a = np.zeros(n - 1)
    c = np.zeros(n - 1)
    d = np.zeros(n - 1)
    h = np.zeros(n - 1)

    for i in range(n - 1):
        a[i] = y[i]
        h[i] = x[i + 1] - x[i]  # length of interval
    # B * x = coef
    # we find x
    B = np.zeros((n, n))
    B[0][0] = 1
    B[n - 1][n - 1] = 1
    for i in range(1, n - 1):
        B[i][i - 1] = 1
        B[i][i] = 4
        B[i][i + 1] = 1
    step = (x[n - 1] - x[0]) / (n - 1)
    coef = np.zeros((n, 1))
    coef[0] = left_d
    coef[-1] = right_d
    for i in range(1, n - 1):
        coef[i] = (3 / step) * (y[i + 1] - y[i - 1])

    b = solve_gauss(B, coef)
    for i in range(n - 1):
        c[i] = (3 / h[i] ** 2) * (y[i + 1] - y[i]) - (b[i + 1] + 2 * b[i]) / h[i]
        d[i] = (-2 / h[i] ** 3) * (y[i + 1] - y[i]) + (1 / h[i] ** 2) * (b[i + 1] + b[i])

    sol = np.zeros(len(points))
    for i in range(len(points) - 1):
        for j in range(n - 1):
            if x[j] <= points[i] < x[j + 1]:
                sol[i] = a[j] + b[j] * (points[i] - x[j]) + c[j] * (points[i] - x[j]) ** 2 + d[j] * (points[i] - x[j]) ** 3
                break
    # sol[len(points) - 1] = g(points[-1])
    return sol



def ex3():
    f = lambda x: 4 * np.sin(2 * x) - 4 * np.cos(4 * x) - 18.87 * x
    f_d = lambda x: 16 * np.sin(4 * x) + 8 * np.cos(2 * x)
    left = -np.pi
    right = np.pi
    left_d = f_d(left)
    right_d = f_d(right)


if __name__ == '__main__':
    # ex1()
    ex2()
    # ex3()
