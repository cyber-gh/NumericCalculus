import numpy as np
import functools
import copy
import math

"""
    Calculates the determinant of a matrix recursively
"""


def determinant(matrix):
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


"""
    first non zero element on the column
"""


def nonzero_pivot_chooser(matrix):
    first_column = matrix[:, 0]
    return (first_column != 0).argmax(axis=0), 0


"""
    # takes the pivot as the first element
    # use this when the matrix has already been reduced once and you
    # know for certain the diagonal is not zero
"""


def first_element_chooser(matrix):
    return 0, 0


"""
    takes the position of the maximum by absolute value
"""


def max_submatrix_chooser(matrix):
    pos = 0, 0
    mx = matrix[0][0]
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1] - 1):
            if abs(matrix[i][j]) > mx:
                mx = abs(matrix[i][j])
                pos = i, j
    # print(mx, pos)
    return pos


"""
    takes the maximum on the current column
"""


def max_column_chooser(matrix):
    first_column = matrix[:, 0]
    return abs(first_column).argmax(axis=0), 0


"""
    Gauss elimination, works on not only on square matrix
    pivot chooser is the function that chooses the pivot, I defined a couple above
    down indicates if it zeros above the main diagonal or below
    returns the modified matrix, the ids of the swapped rows and the ids of the swapped columns
"""


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

        # print("pivot ales ", pivot)
        # print("pivot a fost pe pozitia ", i,j)

        # print("before swap")
        # nice_print(matrix)
        row_swaps.append((idx, i))
        column_swaps.append((idx, j))

        # swap the rows and columns if needed
        matrix[:, [idx, j]] = matrix[:, [j, idx]]
        matrix[[idx, i]] = matrix[[i, idx]]
        # print("after swap")
        # nice_print(matrix)

        # determinate the direction
        way = range(idx + 1, n) if down else range(0, idx)
        for nxt in way:
            # print("nxt = ",matrix[nxt][idx])
            if matrix[nxt][idx] == 0:
                continue
            inverse = -matrix[nxt][idx] / pivot
            # print("inverse pivot = ", inverse)
            matrix[nxt] = np.add(inverse * matrix[idx], matrix[nxt])

        # print("final")
        # nice_print(matrix)
    return matrix, row_swaps, column_swaps


"""
    Prints a matrix nicely formatted
"""


def nice_print(converted):
    print("---------------------")
    for row in converted:
        for el in row:
            print("{:4.3f} ".format(el), end="")
        print()

    print("---------------------")


"""
    Solves a system of equations using gauss method, pivot chooser is customizable
    First 0 out below the diagonal
    Next 0 out above the diagonal
    Last, make the diagonal elements equal to 1
    The answer will be in the last column
"""


def solve_gauss(a, b, pivot_chooser=nonzero_pivot_chooser):
    converted, row_swaps, column_swaps = gauss_elimination(np.c_[a, b], pivot_chooser)
    converted, _, _ = gauss_elimination(converted, first_element_chooser, down=False)
    converted = simplify_diagonal(converted)
    solutions = converted[:, -1]

    # make the appropriate swaps on the solution if we swapped columns while running the gauss elimination
    for i, j in column_swaps:
        solutions[i], solutions[j] = solutions[j], solutions[i]

    return solutions


# a x = b
def ex1():
    a = np.asarray([
        [0, 4, 6, -3],
        [7, -3, -6, -7],
        [6, 9, 1, -7],
        [-3, 1, 8, 3]
    ]).astype(np.float32)

    if determinant(a) != 0:
        print("Sistemul are o solutie unica")
    else:
        print("Sistem nu are solutie unica, Oprim aici")
        return

    b = np.asarray([42, -81, 35, 71]).astype(np.float32)
    solutions = solve_gauss(a, b, max_column_chooser)

    print("Ex 1")
    print("Solutia sistemului")
    print(solutions)

    # nr = converted.shape[0]
    # m = nr
    # solutions = []
    #
    # for idx in range(0, nr):
    #     solutions.append(converted[idx][m] / converted[idx][idx])
    #
    # solutions = list(
    #     map(lambda x: x[0], sorted(zip(solutions, new_order), key=functools.cmp_to_key(lambda x1, x2: x1[0] - x2[0]))))
    #
    # print(solutions)


"""
    Modifies the matrix so the diagonal is 1
"""


def simplify_diagonal(matrix):
    n = matrix.shape[0]
    for idx in range(0, n):
        el = matrix[idx][idx]
        matrix[idx] = (1 / el) * matrix[idx]
        if matrix[idx][idx] < 0:
            matrix[idx] = (-1) * matrix[idx]

    return matrix


"""
    Transforms a matrix to identity matrix
    0 out below the main diagonal
    0 out above the main diagonal
    make the diagonal elements equal to 1
"""


def reduce_matrix(matrix):
    inversed, _, _ = gauss_elimination(matrix, nonzero_pivot_chooser, down=True)
    inversed, _, _ = gauss_elimination(inversed, nonzero_pivot_chooser, down=False)
    inversed = simplify_diagonal(inversed)

    return inversed


def ex2():
    matrixb = np.asarray([
        [0, -5, -9, -2],
        [-2, 2, -2, 9],
        [-1, -2, 2, -5],
        [-3, 7, 5, -2]
    ]).astype(np.float32)
    n = matrixb.shape[0]

    # append the identity matrix to the left of the current one
    identity = np.asarray([[1 if x == y else 0 for x in range(0, n)] for y in range(0, n)])
    combined = np.c_[matrixb, identity]

    # apply the transformation on whole matrix, the result will be in the second half
    inversed = reduce_matrix(combined)[:, n:]

    print("Ex 2")
    print("Inversa matricii")
    nice_print(inversed)


"""
    Alternate method for LU decomposition, shortcut
"""


def doolitle_decomposition(matrix):
    n = matrix.shape[0]
    lower = np.empty_like(matrix)
    lower[:] = 0
    lower = lower.astype(np.float32)

    upper = np.empty_like(matrix)
    upper[:] = 0
    upper = upper.astype(np.float32)
    for j in range(n):
        lower[j][j] = 1.0

        for i in range(j + 1):
            s1 = sum(upper[k][j] * lower[i][k] for k in range(i))
            upper[i][j] = matrix[i][j] - s1

        for i in range(j, n):
            s2 = sum(upper[k][j] * lower[i][k] for k in range(j))
            lower[i][j] = (matrix[i][j] - s2) / upper[j][j]

    return lower, upper


"""
    LU decomposition standard method
    returns lower, upper swaps(tuple of all the indexed of swapped rows needed for rearranging the matrix)
"""


def lu_decomposition(matrix):
    n = matrix.shape[0]

    # make sure the diagonal doesn't contain any 0 elements
    swaps = []
    for idx in range(n):
        i, j = nonzero_pivot_chooser(matrix[idx:, idx:])

        i += idx
        j += idx
        pivot = matrix[i][j]
        if pivot == 0:
            raise ValueError("Pivotul e 0")

        swaps.append((i, idx))

        matrix[:, [idx, j]] = matrix[:, [j, idx]]
        matrix[[idx, i]] = matrix[[i, idx]]

    # l1, u1 = doolitle_decomposition(matrix)
    # return l1, u1, swaps

    # apply the gauss method
    lower = np.asarray([[1 if x == y else 0 for x in range(n)] for y in range(n)]).astype(np.float32)
    for idx in range(0, n):

        pivot = matrix[idx][idx]
        if pivot == 0:
            raise ValueError("Pivotul e 0")

        for nxt in range(idx + 1, n):
            if matrix[nxt][idx] == 0:
                continue
            inverse = -matrix[nxt][idx] / pivot
            matrix[nxt] = np.add(inverse * matrix[idx], matrix[nxt])
            lower[nxt][idx] = -inverse  # we complete the lower matrix here, by assigning
    # nice_print(l1)
    # nice_print(lower)
    return lower, matrix, swaps


"""
    Solves system using LU method
    Ax = B
    L U x = B
    U x = y
    L y = b
"""


def solve_lu(a, b):
    lower, upper, swaps = lu_decomposition(copy.deepcopy(a))

    # rearranging the b column based on the row swaps we made while running the LU decomposition
    for i, j in swaps:
        b[i], b[j] = b[j], b[i]

    # we could also use ascending substitution as its a lower triangular matrix, same thing
    # apply the formula above, get the solution
    y = solve_gauss(lower, b, nonzero_pivot_chooser)
    x = solve_gauss(upper, np.asarray(y).astype(np.float32), nonzero_pivot_chooser)

    return x


def ex3():
    ax3 = np.asarray([
        [0, -3, -10, 2],
        [-9, 8, 5, -4],
        [6, 8, -10, 9],
        [-5, -7, 7, -9]
    ]).astype(np.float32)

    if determinant(ax3) != 0:
        print("System has a unique solution")
    else:
        print("No unique solution, we stop here")
        return

    bx3 = np.asarray([-72, 6, 80, -90])
    solutions = solve_lu(copy.deepcopy(ax3), copy.deepcopy(bx3))
    print("Ex 3")
    print("Solution")
    print(solutions)
    # s_test = solve_gauss(copy.deepcopy(ax3), copy.deepcopy(bx3))
    # print(s_test)


""""
    Check if a matrix is symmetric
"""


def is_symmetric(matrix):
    n, m = matrix.shape
    if n != m:
        raise ValueError("Not a square matrix")

    for i in range(n):
        for j in range(m):
            if matrix[i][j] != matrix[j][i]:
                return False

    return True


"""
    Returns cholesky decomposition of a matrix 
    
    L * Transpose(L) = A
    L being a lower triangular matrix
"""


def cholesky_decomposition(matrix):
    n = matrix.shape[0]

    # this checks if the matrix is positive definite
    if not all(determinant(matrix[:idx, :idx]) > 0 for idx in range(1, n)):
        raise ValueError("Matrix is not cholesky decomposable, determinants <= 0")

    if not is_symmetric(matrix):
        raise ValueError("Matrix is not cholesky decomposable, not symmetric")

    lower = np.empty_like(matrix).astype(np.float32)
    lower[:] = 0

    # run the algorithm for determining the L in the formula
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                lower[j][j] = math.sqrt(matrix[j][j] - sum(lower[j][k] ** 2 for k in range(j)))
            else:
                lower[i][j] = (matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))) / lower[j][j]

    # transpose it to get the other matrix
    upper = np.asarray(
        [[lower[j][i] for j in range(n)] for i in range(n)],
        dtype=np.float32
    )

    return lower, upper


def ex4():
    c = np.asarray([
        [64, -24, -32, 24],
        [-24, 58, -23, -44],
        [-32, -23, 122, -32],
        [24, -44, -32, 140]
    ],
        dtype=np.float32
    )

    lower, upper = cholesky_decomposition(c)
    print("Ex 4")
    print("Cholesky decomposition")
    nice_print(lower)
    nice_print(upper)


if __name__ == '__main__':
    ex1()
    ex2()
    ex3()
    ex4()
    pass
