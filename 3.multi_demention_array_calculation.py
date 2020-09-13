import numpy as np

# 1차원 배열

A = np.array([1, 2, 3, 4])

print("numpy array : " + A) #일반 출력

print("dimention array : " + np.ndim(A)) #몇차원 배열이냐?

print("shape of numpy array : " + A.shape) # 배열의 형상 ----> 튜플 반환(1차원 배열이라도 다차원 배열일때와 동일한 형태로 반환하려고)

print("first element of A.shape : " + A.shape[0])


# 행렬의 곱

A1 = np.array([[1, 2], [3, 4]])

A2 = np.array([[5, 6], [7, 8]])

print(np.dot(A1, A2)) # 내적

