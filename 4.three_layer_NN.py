import numpy as np

# 출력층은 활성화 함수가 다르다. 여기서는 항등 함수를 활성화 함수로 사용한다.

# 출력층의 활성화 함수는 풀고자 하는 문제의 성질에 맞도록 정한다.

# 회귀 : 입력 데이터에서 연속적인 수치를 예측하는 문제 ex). 사진 속 인물의 몸무게 예측 -> 항승 함수

# 분류 : 어느 데이터가 어느 클래스에 속하는지 판별하는 문제 ex). 고양이 vs 개
# - 2 클래스 분류 : 시그모이드 함수
# - 다중 클래스 분류 : 소프트 맥스 함수


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_function(x):
    return x


def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


# 신호가 순방향 (입력에서 출력 방향)으로 전달됨(즉 순전파)을 의미한다.
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)