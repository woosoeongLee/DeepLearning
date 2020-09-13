import numpy as np

# b (bias, 편향) -> 뉴런이 얼마나 쉽게 활성화 되느냐?
# w (wieght, 가중치) -> 각 신호의 영향력 제어
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    result = np.sum(w*x) + b
    if result <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    result = np.sum(w*x) + b
    if result <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    result = np.sum(x*w) + b
    if result <= 0:
        return 0
    else:
        return 1

#XOR 게이트로 단층 퍼셉트론을 표현할 수 없다.
#단층 퍼셉트론은 직선형 영역만 표현할 수 있고, 다층 퍼셉트론은 비선형 영역도 표현할 수 있다.
#다층 퍼셉트론은 (이론상) 컴퓨터를 표현할 수 있다.
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y