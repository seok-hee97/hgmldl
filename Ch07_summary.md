# 딥러닝(DL)
# [07-1]인공 신경망

- 핵심키워드
  - 인공 신경망
  - 텐서플로
  - 밀집층
  - 원-핫 인코딩


```python
# 실행마다 동일한 결과를 얻기 위해 케라스에 랜덤 시드를 사용하고 텐서플로 연산을 결정적으로 만듭니다. 
import tensorflow as tf
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

# https://runebook.dev/ko/docs/tensorflow/config/experimental/enable_op_determinism
```

## 패션 MNIST
: 머신 러닝 배울 때 많이 사용하는 데이터 셋

keras 라이브러리에 데이터 셋 있음(너무 유명해서) load해서 사용

ex] keras.datasets.fashion_mnist.load_data()


```python
from tensorflow import keras
#load mnist data used by keras lib
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
```

train_set 데이터 크기 확인


```python
print(train_input.shape, train_target.shape)
```

    (60000, 28, 28) (60000,)


test_set 데이터 크기 확인


```python
print(test_input.shape, test_target.shape)
```

    (10000, 28, 28) (10000,)


6장에서 matplotlib 라이브러리 사용해서 과일 출력했던 것처럼  
훈련 데이터에서 몇 개의 샘플을 그림으로 출력


```python
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 10, figsize=(10,10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()
```


    
![7-1_9_0](https://github.com/piolink/pyree/assets/98581131/1a78fa3d-ae80-4355-80e9-4219cf423c73)
    


리스트 내포 : List Comprehension(리스트 컴프리헨션)

리스트 안에 표현식(계산식), for문, if문 한줄에 넣어서 새로운 리스트를 만드는 것

https://coding-kindergarten.tistory.com/165

파이썬의 리스트 컴프리헨션을 사용해서 10개의 샘플의 타깃값을 리스트로 만든 후에 출력


```python
print([train_target[i] for i in range(10)])
```

    [9, 0, 0, 3, 0, 2, 7, 2, 5, 5]


train_target 확인  
np.unique() 고유값 확인 메서드(고유값 -> 중복없음)  
return_counts =True 고유값(레이블)당 데이터 카운트 개수 반환  

![fashion_label](https://github.com/piolink/pyree/assets/98581131/08de5468-d0a0-4c35-95ef-6e0c4a7d26b8)



```python
import numpy as np

print(np.unique(train_target, return_counts=True))
```

    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000]))


#각 레이블당 샘플이 6000개씩 들어있는 걸 알 수 있음

## 로지스틱 회귀로 패션 아이템 분류하기
: 훈련 샘플이 60000개나 되기 때문에 전체 데이터를 한꺼번에 사용하여 훈련하는 것보다
샘플을 하나씩 꺼내서 모델을 훈련하는 방법이 효율적으로 보임

패션 MNIST의 경우 각 픽셀 0-255 사이의 정수값 가짐

255 -> 0~1 사이의 값을 정규화(normalization)

양수 값으로 이루어진 이미지를 전처리할 때 널리 사용하는 방법

정규화 하는 이유
스케일 정리(데이터별로 단위값 다르고 값이 너무 클 때)  
(reson 머신러닝에서 scale이 큰 feature의 영향이 비대해지는 것을 방지)

https://velog.io/@cbkyeong/ML%EC%A0%95%EA%B7%9C%ED%99%94normalization%EC%99%80-%ED%91%9C%EC%A4%80%ED%99%94standardization%EB%8A%94-%EC%99%9C-%ED%95%98%EB%8A%94%EA%B1%B8%EA%B9%8C

>4장에서 보았듯이 SGDClassfier 2차원 입력을 다루지 못하기 때문에  
>각 샘플을 1차원 배열로 만들어야함(SGDClassfier 4장, reshape()메서드 6장 참고)


```python
#정규화
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
```

reshpe() 메서드의 두번째 매개변수 28*28 이미지 크기에 맞게 지정해서  
첫번째 차원(샘플 개수)은 변하지 않고 원본 데이터의 두번째, 세번째 차원이 1차원으로 합쳐짐 
 
reshape()메서드 설명:https://yganalyst.github.io/data_handling/memo_5/

변환된 train_scaled 크기 확인


```python
print(train_scaled.shape)
```

    (60000, 784)



```python
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
import numpy as np

sc = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)

scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))

#warning 뜨는 이유 max_iter 학습 반복횟수가 적어서
#max_iter = 100으로 지정하면 해결
```

    0.84185


![logistic_model](https://github.com/piolink/pyree/assets/98581131/38050d63-b599-4d8b-9e80-7ab760b22f96)


## 인공신경망(ANN, artificial neural network)

딥러닝??  
딥러닝은 인공 신경망과 거의 동의어로 사용되는 경우가 많다.  
혹은 심층 신경망(deep nural network, DNN) 딥러닝  
심층 신경망은 다음절에서 보겠지만 여러개의 층을 가진 신경망

![keras_model](https://github.com/piolink/pyree/assets/98581131/79cd6cba-16f0-4d01-a5a3-e29b1cc3491e)

### 텐서플로와 케라스

- 텐서플로(tensorflow)
구글이 2015년 11월 오픈소르로 공개한 딥러닝 라이브러리

- 케라스(keras)
텐서플로의 고수준 API


(∴ 텐서플로 = 케라스)


GPU는 벡터와 행렬 연산에 매우 최적화 -> 곳셈과 덧셈이 많이 수행되는 인공 신경망에 많은 도움


```python
import tensorflow as tf
```


```python
from tensorflow import keras
```

## 인공신경망으로 모델 만들기

인공 신경망 교차검증 잘 사용 X 검증 세트를 별도로 덜어내어 사용  

[이유]
1. 딥러닝 분야의 데이터셋은 충분히 크기 때문에 검증 점수가 안정적
2. 교차 검증을 수행하기에는 훈련 시간이 너무 오래 걸림


```python
from sklearn.model_selection import train_test_split

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
```

훈련세트 크기 확인


```python
print(train_scaled.shape, train_target.shape)
```

    (48000, 784) (48000,)


검증세트 크기 확인


```python
print(val_scaled.shape, val_target.shape)
```

    (12000, 784) (12000,)


keras.layers 패키지 안에는 다양한 층이 준비

가장 기본이 되는 층은 밀집층(dense layer)


keras.layers.Dense 활용해서 층을 만듬


```python
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
# 10 : 뉴런 개수
# activation  뉴런의 출력에 적용할 함수 <- '활성화 함수'라고 부름
```

이제 밀집층을 가진  신경망 모델을 만들자.  
keras.Sequential 클래스 사용


```python
model = keras.Sequential(dense)
```

    Metal device set to: Apple M1
    
    systemMemory: 16.00 GB
    maxCacheSize: 5.33 GB
    


## 인공신경망으로 패션 아이템 분류하기

케라스 모델을 훈련하기 전에 설정 단계

model.complie() 메서드에서 수행

꼭 지정해야할 것이 손실함수의 종류

loss : 손실함수 지정  
metric : 훈련과정에서 계산하고 싶은 측정값



- 이진 분류 : loss = 'binary_crossentropy'
- 다중 분류 : loss = 'categorical_crossentropy'

원-핫 인코딩(one-hot encoding).  
: 타깃값을 해당 클래스만 1이고 나머지는 모두 0인 배열로 만드는 것

다중 분류에서 크로스 엔트로피 손실함수를 사용하러면, 0 ,1, 2  같이 정수로 된 타기값을 원

하지만 텐서플로에서는 정수로 된 타깃값을 원-핫 인코딩으로 바꾸지 않고 그냥 사용 가능

정수로된 타깃값을 사용해 크로스 엔트로피 손실을 계산하는 것이 바로 'sparse_categorical_crossentropy'
(뺵빽한 배열 말고 정수값 하나만 사용한다느 뜻에서 sparse(희소)라는 이름 붙은것 같다)


```python
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
```


```python
print(train_target[:10])
```

    [7 3 5 8 6 9 3 3 9 9]


모델 훈련 model.fit


```python
model.fit(train_scaled, train_target, epochs=5)
```

    Epoch 1/5


    2023-07-18 23:52:23.912369: W tensorflow/tsl/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz


    1500/1500 [==============================] - 10s 6ms/step - loss: 0.6080 - accuracy: 0.7947
    Epoch 2/5
    1500/1500 [==============================] - 9s 6ms/step - loss: 0.4785 - accuracy: 0.8386
    Epoch 3/5
    1500/1500 [==============================] - 9s 6ms/step - loss: 0.4565 - accuracy: 0.8477
    Epoch 4/5
    1500/1500 [==============================] - 9s 6ms/step - loss: 0.4435 - accuracy: 0.8535
    Epoch 5/5
    1500/1500 [==============================] - 9s 6ms/step - loss: 0.4356 - accuracy: 0.8560





    <keras.callbacks.History at 0x17184a820>




```python
model.evaluate(val_scaled, val_target)
```

    375/375 [==============================] - 2s 4ms/step - loss: 0.4598 - accuracy: 0.8460





    [0.4598352015018463, 0.8460000157356262]



검증 세트의 점수가 훈련세트보다 조금 낮은 것이 일반적
# [7-2]심층 신경망

> 인공 신경망에 층을 여러 개 추가하여 패션 MINST 데이터 셋을 분류하면서 케라스로 심층 신경망을 만든느 방법을 자세히 배웁니다

- 핵심 키워드
  - 심층 신경망
  - 렐루 함수
  - 옵티마이저


```python
# 실행마다 동일한 결과를 얻기 위해 케라스에 랜덤 시드를 사용하고 텐서플로 연산을 결정적으로 만듭니다. 
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
```

## 2개의 층

![2_layers](https://github.com/piolink/pyree/assets/98581131/c8d7141d-b3c0-490d-88f8-dfbede73b5d8)

[출력층(output_layer)]
- 분류 문제는 클래스에 대한 확률을 출력하기 위해 활성화 함수 사용
- 회귀 출력은 임의의 어떤 숫자이므로 확설화 함수를 적용할 필요가 없다.
즉 출력층의 선형 방정식의 계산을 그대로 출력

케라스 API를 사용해서 패션 MNIST 테이터셋 로드


```python
from tensorflow import keras

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
    29515/29515 [==============================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
    26421880/26421880 [==============================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
    5148/5148 [==============================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
    4422102/4422102 [==============================] - 0s 0us/step


train_test_split() 함수로 훈련 세트와 검증 세트 Split!!!


```python
from sklearn.model_selection import train_test_split

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
```

dense1, dense2 층 2개 만듬

활성화 함수중 sigmoid 함수와 softmax 함수

[활성화 함수 종류]  
https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=handuelly&logNo=221824080339


```python
dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
dense2 = keras.layers.Dense(10, activation='softmax')
```

## 심층 신경망 만들기


```python
model = keras.Sequential([dense1, dense2])
```

케라스 모델의 summary() 메서드 호출하면 층에 대한 유용한 정보 get!!

층마다 층 이름, 클래스, 출력크기, 모델 파라미터 개수 추력

[Output Shape]  

(None, 100)     100개 출력  
(None, 10)      10 개 출력  
None 부분 배치 사이즈 설정부분  
케라스의 기본 미니배치 크기는 32개  
fit() 메서드에서 batch_size 매개변수로 바꿀 수 있음  

따로 설정하지않으면 유연하게 대응함  

[Param]

#매개변수 각층에 있는 가중치와 절편



```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 100)               78500     
                                                                     
     dense_1 (Dense)             (None, 10)                1010      
                                                                     
    =================================================================
    Total params: 79,510
    Trainable params: 79,510
    Non-trainable params: 0
    _________________________________________________________________


## 층을 추가하는 다른 방법

[방법 1] Sequential 클래스에 생성자 안에 바로 Dense 클래스의 객체를 만드는 경우 많음


```python
model = keras.Sequential([
    keras.layers.Dense(100, activation='sigmoid', input_shape=(784,), name='hidden'),
    keras.layers.Dense(10, activation='softmax', name='output')
], name='패션 MNIST 모델')
```

모델의 이름과 달리 층의 이름은 반드시 영문


```python
model.summary()
```

    Model: "패션 MNIST 모델"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     hidden (Dense)              (None, 100)               78500     
                                                                     
     output (Dense)              (None, 10)                1010      
                                                                     
    =================================================================
    Total params: 79,510
    Trainable params: 79,510
    Non-trainable params: 0
    _________________________________________________________________


[방법 2] Sequential() 클래스의 객체를 만드로 이 객체의 add() 메서드를 호출하여 층을 추가

한눈에 추가되는 층을 볼 수 있고 프로그램 실행 시 동적으로 층을 선택하여 추가 가능


```python
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))
```


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_2 (Dense)             (None, 100)               78500     
                                                                     
     dense_3 (Dense)             (None, 10)                1010      
                                                                     
    =================================================================
    Total params: 79,510
    Trainable params: 79,510
    Non-trainable params: 0
    _________________________________________________________________



```python
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

model.fit(train_scaled, train_target, epochs=5)
```

    Epoch 1/5
    1500/1500 [==============================] - 9s 3ms/step - loss: 0.5710 - accuracy: 0.8064
    Epoch 2/5
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.4132 - accuracy: 0.8509
    Epoch 3/5
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.3776 - accuracy: 0.8646
    Epoch 4/5
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3530 - accuracy: 0.8732
    Epoch 5/5
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3344 - accuracy: 0.8782





    <keras.callbacks.History at 0x7fb97c311370>



## 렐루(ReLU) 활성화 함수
기존의 시그모이드 활성화 함수의 단점을 개선  
시그모이드 함수 단점: 양쪽 끝으로 갈수록 그래프가 누워있기 때문에 올바른 출력을 만드는데 신속하게 대응하지 못함

max(0,z)로 표현도 가능

전에 학습할 때 패션 MNIST 이미지파일(28*28) 때문에 인공 신경망에 주입하기 위해  
numpy.reshape() 메서드 사용해서 1차원 배열로 만듬

케라스에서는 Flatten 층을 제공


[Flatten layer]
: Flatten 클래스는 배치 차원을 제외하고 나머지 입력 차원을 모두 일렬로 펼치는 역할만 함  
(성능에 기여하는 바는 없음)


```python
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
```

flatten 층을 신경망 모델에 추가하면 입력값의 차원을 짐작할 수있는 것이 장점


```python
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten (Flatten)           (None, 784)               0         
                                                                     
     dense_4 (Dense)             (None, 100)               78500     
                                                                     
     dense_5 (Dense)             (None, 10)                1010      
                                                                     
    =================================================================
    Total params: 79,510
    Trainable params: 79,510
    Non-trainable params: 0
    _________________________________________________________________



```python
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
```


```python
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

model.fit(train_scaled, train_target, epochs=5)
```

    Epoch 1/5
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.5290 - accuracy: 0.8113
    Epoch 2/5
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3920 - accuracy: 0.8576
    Epoch 3/5
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.3525 - accuracy: 0.8726
    Epoch 4/5
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.3301 - accuracy: 0.8821
    Epoch 5/5
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3141 - accuracy: 0.8867





    <keras.callbacks.History at 0x7fb97c016d00>




```python
model.evaluate(val_scaled, val_target)
```

    375/375 [==============================] - 1s 2ms/step - loss: 0.3683 - accuracy: 0.8726





    [0.3683287501335144, 0.8725833296775818]



## 옵티마이저(Optimizer)

: 조정해야할 수많은 하이퍼파라미터 중 하나

다양한 종류의 경사 하강법 알고리즘 = 옵티마지어(Optimizer)

SGD 옵티마이저를 사용하려면 compile() 메서드의 optimizer 매개변수 'sgd'로 지정

옵티마이저 참고   
https://wikidocs.net/152765

![optimizer](https://github.com/piolink/pyree/assets/98581131/07143b21-6189-412f-a4a4-774265942f7d)


```python
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics='accuracy')
```

 sgd 옵티마지어는 tensorflow.keras.optimziers 패키지 아래 SGD 클래스로 구현되어 있음  
'sgd' 문자열 이 클래스의 기본 설정 매개변수로 생성한 객체와 동일

위에 코드와 동일(다른 표현)


```python
sgd = keras.optimizers.SGD()
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics='accuracy')
```

SGD 클래스의 학습률 기본 값이 0.01, learning_rate 매개변수 지정해서 사용


```python
sgd = keras.optimizers.SGD(learning_rate=0.1)
```

![optimizer_2](https://github.com/piolink/pyree/assets/98581131/7fde344e-59c6-4868-86ae-13b559188755)

기본 경사 하강법 옵티마이저 모두 SGD 클래스 제공. SGD 클래스의 momentum 매개변수 기본값 0  
이르 0보다 큰 값으로 지정하면 마치 이전의 그레이디언트를 가속도처럼 사용하는 모멘텀 최적화(momentum optimization).   
보통 momentum 매개변수는 0.9 이상 지정

SGD 클래스의 netsterov 매개변수를 기본값 False에서 True로 바꾸면.  
 네스테로프 모멘텀 최적화(netsterov momentum optimization)(또는 네스테로프 가속 경사)를 사용
더 나은 성능 제공

이런 학습률을 [적응적 학습률(adaptive learning rate)]. 
이런 방식들은 학습률 매개변수를 튜닝하는 수고를 덜 수 있음

적응적 학습률(adaptive learning rate)를 사용하는 대표적인 옵티마이저  
ex] adagrad, rmsprop


```python
sgd = keras.optimizers.SGD(momentum=0.9, nesterov=True)
```


```python
adagrad = keras.optimizers.Adagrad()
model.compile(optimizer=adagrad, loss='sparse_categorical_crossentropy', metrics='accuracy')
```


```python
rmsprop = keras.optimizers.RMSprop()
model.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy', metrics='accuracy')
```


```python
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
```


```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

model.fit(train_scaled, train_target, epochs=5)
```

    Epoch 1/5
    1500/1500 [==============================] - 5s 3ms/step - loss: 0.5263 - accuracy: 0.8157
    Epoch 2/5
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3970 - accuracy: 0.8580
    Epoch 3/5
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3556 - accuracy: 0.8701
    Epoch 4/5
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3277 - accuracy: 0.8802
    Epoch 5/5
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3087 - accuracy: 0.8856





    <keras.callbacks.History at 0x7fb98b8ef190>




```python
model.evaluate(val_scaled, val_target)
```

    375/375 [==============================] - 1s 3ms/step - loss: 0.3526 - accuracy: 0.8733





    [0.3525600731372833, 0.8732500076293945]



> 환경마다 조금씩 차이가 있을 수 있지만 여기서는 기본 RMSprop보다 조금 나은 성능을 냅니다.
# [7-3]신경망 모델 훈련

> 인공 신경망 모델을 훈련하는 모범 사례와 필요한 도구들을 살펴보겠습니다.
> 이런 도구들을 다뤄보면서 텐서플로와 케라스 API에 더 익숙해 질 것입니다.

- 핵심 키워드
  - 드롭 아웃
  - 콜백
  - 조기종료


```python
# 실행마다 동일한 결과를 얻기 위해 케라스에 랜덤 시드를 사용하고 텐서플로 연산을 결정적으로 만듭니다. 
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
```

## 손실 곡선

패션 MNIST 데이터셋을 적재하고 훈련세트와 검증 세트로 나눔


```python
from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
```

모델을 만드는 함수 간단하게 만듬


```python
def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model
```


```python
model = model_fn()

model.summary()
```

    Metal device set to: Apple M1
    
    systemMemory: 16.00 GB
    maxCacheSize: 5.33 GB
    
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten (Flatten)           (None, 784)               0         
                                                                     
     dense (Dense)               (None, 100)               78500     
                                                                     
     dense_1 (Dense)             (None, 10)                1010      
                                                                     
    =================================================================
    Total params: 79,510
    Trainable params: 79,510
    Non-trainable params: 0
    _________________________________________________________________



```python
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

history = model.fit(train_scaled, train_target, epochs=5, verbose=0)
```

    2023-07-19 03:51:49.509303: W tensorflow/tsl/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz


[+ verbos = 0 ?]
verbose 매개변수 훈련 과정 출력을 조절  
기본값은 1로 이전 절처럼 에포크마다 진행 막대화 함께 손실등의 지표 출력  
2로 바꾸면 진행 막대를 빼고 출력  
verbose = 0으로 하면 훈련 과정 출력 X

history 객체에 훈력 측정값이 담겨 있는 history 딕셔너리가 들어 있음


```python
print(history.history.keys())
```

    dict_keys(['loss', 'accuracy'])



```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```


    
![7-3_12_0](https://github.com/piolink/pyree/assets/98581131/83f75365-add3-48a0-8b42-06fdf91f2eb2)
    



```python
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
```


    
![7-3_13_0](https://github.com/piolink/pyree/assets/98581131/033b8c56-7d53-4f7e-aedd-f1070571ccb3)
    


모델을 훈련하고 fit() 메서드의 결과를 history  변수에 담아봄


```python
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

history = model.fit(train_scaled, train_target, epochs=20, verbose=0)
```


```python
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```


    
![7-3_16_0](https://github.com/piolink/pyree/assets/98581131/e1e9c7ba-1574-4608-93d2-14a7315933aa)
    


## 검증 손실

에포크마다 검증 손실을 계산하기 위해 케라스 모델의 fit() 메서드에 검증 데이터 전달 


```python
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

history = model.fit(train_scaled, train_target, epochs=20, verbose=0, 
                    validation_data=(val_scaled, val_target))
```


```python
print(history.history.keys())
```

    dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])



```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```


    
![7-3_21_0](https://github.com/piolink/pyree/assets/98581131/341b3918-9089-4209-8fbb-6997a0d96b57)
    



```python
model = model_fn()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics='accuracy')

history = model.fit(train_scaled, train_target, epochs=20, verbose=0, 
                    validation_data=(val_scaled, val_target))
```


```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```


    
![7-3_23_0](https://github.com/piolink/pyree/assets/98581131/820ac6c2-a9a0-466e-bff2-2593e288ca2f)
    


## 드롭아웃(dropout)
: 드롭아웃 구글과 함께 제프리 힌턴이  소개
훈련과정에서 층에 있는 뉴런을 랜덤하게 꺼서(즉 뉴런의 출력을 0으로 만들어)  
과대적합을 막는 방법

![dropout](https://github.com/piolink/pyree/assets/98581131/83b45eb6-2b0c-4ca6-ae7d-2ee2e43c0337)


```python
#위에서 정의한 함수
def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model


model = model_fn(keras.layers.Dropout(0.3))

model.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_4 (Flatten)         (None, 784)               0         
                                                                     
     dense_8 (Dense)             (None, 100)               78500     
                                                                     
     dropout (Dropout)           (None, 100)               0         
                                                                     
     dense_9 (Dense)             (None, 10)                1010      
                                                                     
    =================================================================
    Total params: 79,510
    Trainable params: 79,510
    Non-trainable params: 0
    _________________________________________________________________



```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics='accuracy')

history = model.fit(train_scaled, train_target, epochs=20, verbose=0, 
                    validation_data=(val_scaled, val_target))
```


```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```


    
![7-3_28_0](https://github.com/piolink/pyree/assets/98581131/6681c2aa-16a5-4607-b8d3-a4c20ebb121b)
    


## 모델 저장과 복원


```python
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics='accuracy')

history = model.fit(train_scaled, train_target, epochs=10, verbose=0, 
                    validation_data=(val_scaled, val_target))
```

케라스 모델은 훈련된 모델의 파라미터를 저장하는 간편한 save_weight() 메서드 제공

이 메서드는 텐서플로의 체크포인트 포맷을 저장하지만 파일확장자가 '.h5'일 경우 HDF5 포맷으로 저장


```python
model.save_weights('model-weights.h5')
```

모델 구조 + 모델 파라미터 함께 저장 save() 메서드


```python
model.save('model-whole.h5')
```

쉘 명령어 실행할때 ! 사용


```python
!ls -al *.h5
```

    -rw-r--r--  1 jangseokhee  staff  6912280 Jul 13 11:00 best-cnn-model.h5
    -rw-r--r--  1 jangseokhee  staff   333448 Jul 19 04:15 model-weights.h5
    -rw-r--r--  1 jangseokhee  staff   982664 Jul 19 04:15 model-whole.h5


load_weights() 메서드 사용해서 기존의 save_weights()로 저장한 값을 불러서 사용

주의) 메서드로 지정했던 모델과 정확히 같은 구조여야 함  
그렇지 않으면 에러 발생


```python
model = model_fn(keras.layers.Dropout(0.3))

model.load_weights('model-weights.h5')
```

모델의 predict() 메서드 결과에서 가장 큰 값을 고르기 위해 numpy argmax() 함수 사용  
axis = -1 배열의 마지막 차원까지를 의미함


```python
import numpy as np

val_labels = np.argmax(model.predict(val_scaled), axis=-1)
print(np.mean(val_labels == val_target))
```

    375/375 [==============================] - 1s 1ms/step
    0.8775


evaluate() 메서드를 사용해 정확도 출력

먼저 compile() 메서드를 실행해야함

(학습을 해야 정확도를 알 수 있음)


```python
model = keras.models.load_model('model-whole.h5')

model.evaluate(val_scaled, val_target)
```

    375/375 [==============================] - 1s 2ms/step - loss: 0.3388 - accuracy: 0.8775





    [0.3387581408023834, 0.8774999976158142]



load_mode() 하면 파라미터 뿐만 아니라 모델 구조와 옵티마이저 상태까지 모두 복원  
그래서 evaluate() 메서드 사용가능  
(텐서플로 2.3에서는 버그 문제로 compile() 하고 evaluate())

## 콜백(Callback)
: 훈련 과정 중간에 어떤 작업을 수행할수 있게 하는 객체  
keras.callbacks 패키지 아라에 있는 클래스

fit() 메서드의 callbacks  매개변수에 리스트로 전달하여 사용  
여기서 ModelCheckpoint 콜백은 기본적으로 에포크마다 모델을 저장  
save_best_only = True 매개변수를 지정하여 가장 낮은 검증 점수(좋은 모델)를 만드는 모델을 저장  
'best-model.h5' 저장될 파일이름


```python
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics='accuracy')

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5', 
                                                save_best_only=True)

model.fit(train_scaled, train_target, epochs=20, verbose=0, 
          validation_data=(val_scaled, val_target),
          callbacks=[checkpoint_cb])
```




    <keras.callbacks.History at 0x2f978dd30>




```python
model = keras.models.load_model('best-model.h5')

model.evaluate(val_scaled, val_target)
```

    375/375 [==============================] - 2s 4ms/step - loss: 0.3154 - accuracy: 0.8878





    [0.31535962224006653, 0.8877500295639038]



#### [조기종료(early stopping)]
: 과대적합이 시작되기 전에 훈련을 미리 중지

조기종료는 훈련 에포크 횟수를 제한하는 역할이지만 모델이 과대적합되는 것을 막아 구지 때문에 규제 방법 중 하나라고 생각할 수 도 있음

EarlyStopping 콜백을  ModelCheckpoint 콜백과 함께 사용하면 가장 낮은 검증 손실의 모델을 파일에 저장하고  
검증 손실이 다시 상승할 때 훈열을 중지할 수 있다.


```python
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics='accuracy')

#ModelCheckpoint 콜백
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5', 
                                                save_best_only=True)

#EarlyStopping 콜백
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
                                                  restore_best_weights=True)
#restore_best_weight <- 훈련을 줒징한 다음 현재 모델의 파라미터를 최상의 파라미터로 되돌림

history = model.fit(train_scaled, train_target, epochs=20, verbose=0, 
                    validation_data=(val_scaled, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

훈련을 마치고 나면 몇번째 에포크에서 훈련이 중지되었는지 early_stopping_cb 객체의 Stopped_epoch 속성에서 확인 가능


```python
print(early_stopping_cb.stopped_epoch)
```

    13



```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```


    
![7-3_53_0](https://github.com/piolink/pyree/assets/98581131/6da39391-b59e-43e3-ab08-bfbcc26ce059)
    


조기종료로 얻은 모델을 사용해 검증 세트에 대한 성능 확인


```python
model.evaluate(val_scaled, val_target)
```

      1/375 [..............................] - ETA: 20s - loss: 0.2539 - accuracy: 0.8750375/375 [==============================] - 2s 4ms/step - loss: 0.3166 - accuracy: 0.8863





    [0.31660589575767517, 0.8863333463668823]




```python

```
