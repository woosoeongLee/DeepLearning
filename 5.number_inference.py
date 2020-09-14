# 일단 학습된 매개변수를 사용한다. (학습 과정 생략)
# 추론 과정만 구현(순전파)

from dataset.mnist import load_mnist
import numpy as np
from PIL import Image


# load_mnist 함수는 읽은 MNIST 데이터를 "(훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)" 형식으로 반환
# 첫 번째 인자(normalize) : 입력 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다. False면 0~255 사이의 값 유지
# 두 번째 인자(flatten) : 입력 이미지를 1차원 배열로 만들지 결정. False 면 입력 이미지의 픽셀은 원래 값 그대로 0~255 사이의 값 유지,
# True 면 784개의 원소로 이루어진 1차원 배열로 저장
# 세 번째 인자(one-hot-label) : 원-핫 인코딩 형태로 저장할지 결정. 정답을 뜻하는 원소만 1, 나머지는 0인 형태로 저장하는 기법. False 면 숫자 형태로 저장.


# flatten = True 로 설정해 읽어 들인 이미지는 1차원 넘파이 배열로 저장된다.
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 각 데이터의 형상 출력
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)


def image_show(img):
    pil_img = Image.fromarray(np.uint8(img))  # 넘파이로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환해야 한다.
    pil_img.show()


img = x_train[0]
label = t_train[0]
print("label of trained data ", label)  # 5

print("shape of image : ", img.shape) # (784, )


# 현재 1차원 넘파이 배열 형태이므로 원래 형상인 28 x 28 형태로 다시 변형해야 한다.
img = img.reshape(28, 28)  # 원래 이미지의 모양으로 변형
print(img.shape)

image_show(img)



