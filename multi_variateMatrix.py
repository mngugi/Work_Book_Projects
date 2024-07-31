def matrix_multivariet(num,matrix):
    main = list()
    for i in matrix:
        temp = list()
        for j in i:
            temp.append(j*num)
        main.append(temp)
    return main
        
