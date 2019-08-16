import numpy as np

"""
常用的 function
numpy.random.seed(number) → 控制output 結果
numpy.random.random((size=None)) → 建立一個數值為0到1之間的隨機array
numpy.zeros((shape)) → 建立一個全部為0的矩陣
numpy.matmul(matrix1, matrix2) → 矩陣相乘
numpy.vstack(matrix1, matrix2) → 已列的方式對array或matrix做組合
"""


# Define our target function
def my_function(x):
    # Write your code below!
    result = 0.05*x**2 + 0.8*x
    return result


# Define derivative
epsilon = 0.1
def derivative(f, x):
    # f in here stands for our function, and x is our input
    # I usually set a variable called epsilon, it represents a small number
    # Write your code below!
    h = epsilon
    result = (f(x + h) - f(x)) / h
    return result


def my_function2(X):
    # For me, I tend to use uppercase X to represent a list or a matrix
    # and use lowercase x to represent a single value
    # You can change the variable to whatever that you feel natural
    # just remember that the X here represent a list of two variables x and y
    # in which X[0] represents x and X[1] represents y
    # Write your code below!
    result = 2*X[0]**2 + 3*X[0]*X[1] + 5*X[1]**2
    return result


def partial_derivative(f, X, i):
    # f is our function, and i is simply the index which we are
    # excuting our partial derivative on
    H = X.copy()
    h = epsilon
    H[i] = X[i] + h
    result = (f(H) - f(X)) /h
    return result




