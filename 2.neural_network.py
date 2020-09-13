# 가중치 매개변수의 값을 데이터로부터 자동으로 학습해서 적절한 값으로 초기화 시키기 위한 것.
# 입력층 / 은닉층 / 출력층으로 구성된다.

# h(x) : 입력 신호의 총합을 출력 신호로 변환하는 함수를 활성화 함수라고 한다.
# 활성화 함수는 입력 신호의 총합이 활성화를 일으키는지를 정하는 역할

# a = b + w1x1 + w2x2   <----퍼셉트론에도 있었음 (가중치가 달린 입력 신호와 편향의 총합을 계산)
# y = h(a)              <----최종적으로 y를 출력

# 단층 퍼셉트론 : 단층 퍼셉트론에서 계단 함수를 활성화 함수로 사용한 모델
# 다층 퍼셉트론 : 다층 퍼셉트론에서 신경망(시그모이드 등의 매끈함 함수를 활성화 함수로 사용하는 모델)을 의미

# 활성화 함수로 사용할 수 있는 여러가지 후보중에 퍼셉트론은 계단 함수를 사용하는 것

import numpy as np
import matplotlib.pylab as plt

# 차이점
# 1. 매끄러움의 차이
# 2. 퍼셉트론에서는 뉴런 사이에 0/1이 흐르지만, 신경망에서는 연속적인 실수가 흐른다.

#공통점
# 1. 둘 다 입력이 작을 때의 출력은 0에 가깝고(혹은 0이고), 입력이 커지면 출력이 1에 가까워지는(혹은 1이되는) 구조
# 즉, 입력이 중요하면 큰 값을 출력하고 입력이 중요하지 않으면 작은 값을 출력한다.
# 2. 입력이 아무리 작거나 커도 출력은 0 이상 1 이하이다.

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU 에서 Rectified란 '정류된'이란 뜻.(전기회로쪽 용어)
# x가 0 이하일 때를 차단하여 아무 값도 출력하지 않는(0을 출력하는) 것. 즉 '정류된 선형 함수'

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1) # -5.0에서 5.0 전까지 0.1간격의 넘파이 배열을 생성 [-5.0, -4.9, ... 4.9]

#y = step_function(x)

y = sigmoid(x)
plt.plot(x, y) # x, y축 생성
plt.ylim(-0.1, 1.1)
plt.show()

