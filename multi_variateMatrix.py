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
        
