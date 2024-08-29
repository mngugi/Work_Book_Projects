from sklearn import svm


x = [
    [0.1, 2.3]
    [-1.5, 2.5]
    [2.0, -4.3]    
    
]

y = [0,1,0]


svm.SVC().fit(x,y)