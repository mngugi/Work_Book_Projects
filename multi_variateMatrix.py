def matrix_multivariet(num,matrix):
    main = list()
    for i in matrix:
        temp = list()
        for j in i:
            temp.append(j*num)
        main.append(temp)
    return main
        
def transpose(matrix):
    main - list()
    for i in range(len(matrix[0])):
        temp = list()
        for j in range (len(matrix)):
            temp.append(matrix[j][i])
        main.append(temp)
    return main

def vaector_multivariate(matrix1,matrix2):
    main = list()
    for i in matrix1:
        temp_list = list()
        for j in transpose(matrix2):
            tempt = 0
            for q in range(len(i)):
                temp += i[q]*j[q]
            temp_list.append(temp)
        main.append(temp_list)
    return main    
        
def determinant(matrix):
    if len(matrix) > 2:
        matrix_0 matrix[1::]
        main = list()
        for i range(len(matrix)):
            temp = transpose(matrix_o)
            temp.pop(i)
            temp = transpose(temp)
            main.append(temp)
        det = 0
        for i in range(len(matrix[0])):
            if 1%2 == 0:
                det += matrix[0][i]*det(main[i])
            else:
                det += matrix[0][i]*det(main[i])
        return det        
                
            
    else:
        det = (matrix[0][0]*matrix[1][1]-matrix[1][0]*matrix[0][1])
        return det



def adj(matrix):
    main = list()
    for i in range(len(matrix)):
        append_temp = list()
        for j in range(len(matrix[i]))
        temp = matrix[:]
        temp = transpose(matrix_o)
            temp.pop(i)
            temp = transpose(temp)
            temp.pop(j)
            temp = transpose(temp)
            append_temp.append(det(temp))
            
        main.append(append_temp)
        
    for i in range(len(matrix)):
        for j in range(len(main[i])):
            main[i][j] = (-1**(i+j+2))*main[i][j]
    main = transpose(main)
    return main

    
