def matrix_mult(num, matrix):
    main = []
    for row in matrix:
        main.append([element * num for element in row])
    return main

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def vector_multivariate(matrix1, matrix2):
    result = []
    for row in matrix1:
        result_row = []
        for col in transpose(matrix2):
            result_row.append(sum(x * y for x, y in zip(row, col)))
        result.append(result_row)
    return result

def determinant(matrix):
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
    
    det = 0
    for c in range(len(matrix)):
        submatrix = [row[:c] + row[c+1:] for row in matrix[1:]]
        sign = (-1) ** c
        sub_det = determinant(submatrix)
        det += (sign * matrix[0][c] * sub_det)
    return det

def adj(matrix):
    adjugate = []
    for r in range(len(matrix)):
        adjugate_row = []
        for c in range(len(matrix)):
            submatrix = [row[:c] + row[c+1:] for row in (matrix[:r] + matrix[r+1:])]
            cofactor = ((-1) ** (r + c)) * determinant(submatrix)
            adjugate_row.append(cofactor)
        adjugate.append(adjugate_row)
    return transpose(adjugate)

def inverse(matrix):
    det = determinant(matrix)
    if det == 0:
        return "Matrix is singular and cannot be inverted."
    adjugate_matrix = adj(matrix)
    return matrix_mult(1 / det, adjugate_matrix)

# Sample non-singular matrices for testing
x1 = [
    [4, 7],
    [2, 6]
]

x2 = [
    [3, 0, 2],
    [2, 0, -2],
    [0, 1, 1]
]

# Calculate inverse of x1
inverse_x1 = inverse(x1)

# Calculate inverse of x2
inverse_x2 = inverse(x2)

inverse_x1, inverse_x2
