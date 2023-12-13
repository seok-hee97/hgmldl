# CH04 . 다양한 분류 알고리즘
## CH04-1 . 로지스틱 회귀

### 데이터 준비하기
```python
# 판다스를 사용하여 csv 데이터를 읽어 들입니다.
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
```
<img src="https://github.com/piolink/pyree/assets/90364637/b6227f7e-af91-4f63-a0c0-a6e2f58084f1" width="300" height="200"><br>



species는 타깃 데이터로 사용하고, Weight, Length, Diagonal, Height, Width는 입력 데이터로 사용합니다.
데이터프레임을 넘파이 배열로 바꾸어 저장합니다.
```python
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()
```

### K-최근접 이웃 분류기의 확률 예측
```python
# 데이터 세트 2개 준비
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state = 42)

# 훈련 세트와 데이터 세트를 표준화 전처리한다.
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# k-최근접 이웃으로 훈련한다.
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))
```

### K-최근접 이웃의 다중 분류
다중 분류는 타깃 테이터에 2개 이상의 클래스가 포함된 문제이다. <br><br>

정렬된 타깃의 값은 classes_에 저장되어 있고, predict()를 통해 5개를 예측할 수 있다.
```python
print(kn.classes_)
print(kn.predict(test_scaled[:5]))
```
```
['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
['Perch' 'Smelt' 'Pike' 'Perch' 'Perch']
```

predict_proba()는 클래스별 확률값을 변환한다.
```python
import numpy as np

proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))
```
```
[[0.     0.     1.     0.     0.     0.     0.    ]
 [0.     0.     0.     0.     0.     1.     0.    ]
 [0.     0.     0.     1.     0.     0.     0.    ]
 [0.     0.     0.6667 0.     0.3333 0.     0.    ]
 [0.     0.     0.6667 0.     0.3333 0.     0.    ]]
```
k-최근접 이웃은 3개의 이웃을 사용하기 때문에 0, 1/3, 2/3, 3/3으로 4가지 확률만 존재하기에 더 좋은 확률이 필요하다.<br><br>

### 로지스틱 회귀
선형방정식을 활용한 분류 알고리즘이다.<br>
시그모이드 함수 (로지스틱 함수)와 소프트맥스 함수를 사용한다.<br>


*  시그모이드 함수 [ 로지스틱 함수 ] <br>
  출력한 선형방정식을 0과 1 사이 값으로 나타낸다. <br>
  이진분류일 경우 0.5보다 크면 양성 클래스, 0.5보다 작거나 같으면 음성 클래스이다.<br>
  이중 분류에서 사용한다.<br>

  <img src="https://github.com/piolink/pyree/assets/90364637/6696e2ad-87b8-4a1d-a7d0-630d25a830db" width="400" height="300"><br><br>
```python
import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))

plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()
```
 <img src="https://github.com/piolink/pyree/assets/90364637/061ce42b-a862-42b1-87b3-2fde603d6949" width="400" height="300"><br><br>
  
* 소프트맥스 함수 <br>
  출력한 선형방정식을 0과 1 사이 값으로 나타내고, 전체 합이 1이 되도록 만든다.<br>
  지수 함수를 사용하기 때문에 정규화된 지수 함수라고 부르기도 한다.
  다중 분류에서 사용한다.<br>
  

### 로지스틱 회귀로 이진 분류하기
<br>
블리언 인덱싱은 넘파이 배열 TRUE, FALSE 값을 전달하여 행을 선택할 수 있다.<br>
도미와 빙어의 행을 모두 TRUE로 만들어 적용하면 도미와 빙어 데이터만 골라낼 수 있다 <br>

```python
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
```

아래의 코드는 도미와 빙어의 샘플을 예측하고 확률을 예측하고 속성을 확인할 수 있다.
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5]))
print(lr.classes_)
```

```python
# 가중치와 절편을 출력한다.
print(lr.coef_, lr.intercept_)
# 5개의 샘플을 z 값으로 출력한다.
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

# 시그모이드 함수로 변환한다.
from scipy.special import expit
print(expit(decisions))
```
```
출력값
[[-0.4037798  -0.57620209 -0.66280298 -1.01290277 -0.73168947]] [-2.16155132]
[-6.02927744  3.57123907 -5.26568906 -4.24321775 -6.0607117 ]
[0.00240145 0.97264817 0.00513928 0.01415798 0.00232731]
```

### 로지스틱 회귀로 다중 분류하기

c는 규제를 제어하는 매개변수이다.<br>
c는 alpha와 반대로 작을수록 규제가 커진다.<br>
```python
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
```

```python
print(lr.predict(test_scaled[:5]))
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))
print(lr.classes_)
print(lr.coef_.shape, lr.intercept_.shape)

decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

# 다중 분류에서는 소프트맥스 함수로 변환한다.
from scipy.special import softmax

proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
```
```
출력값
[[0.    0.014 0.841 0.    0.136 0.007 0.003]
 [0.    0.003 0.044 0.    0.007 0.946 0.   ]
 [0.    0.    0.034 0.935 0.015 0.016 0.   ]
 [0.011 0.034 0.306 0.007 0.567 0.    0.076]
 [0.    0.    0.904 0.002 0.089 0.002 0.001]]
```
## CH04-2. 확률적 경사 하강법
점진적인 학습은 가중치(계수)와 절편을 유지하면서 업데이트하는 것을 의미한다. <br>
점진적인 학습 알고리즘은 확률적 경사 하강법이다.

  * 확룰적 경사 하강법<br>
  확률적 경사 하강법은 선형회귀나 로지스틱 회귀처럼 머신러닝 알고리즘은 아니다. <br>
  확률적 경사 하강법은 훈련 세트에서 샘플 하나씩 꺼내 손실 함수의 경사를 따라 최적의 모델을 찾는 알고리즘이다.<br> 

  * 미니배치 경사 하강법<br>
  여러 개의 샘플을 사용해 경사 하강법을 수행하는 방식이다.<br>
  주로 2의 배수를 활용한다.<br>

  * 배치 경사 하강법<br>
  전체 샘플을 사용해 경사 하강법을 수행하는 방식이다. <br>
  전체 데이터를 사용하면 그만큼 컴퓨터 자원을 많이 사용하게 된다. <br>

  * 에포크 <br>
  확률적 경사 하강법에서 훈련세트를 한 번 모두 사용하는 과정을 의미한다.<br>
 
<img src="https://github.com/piolink/pyree/assets/90364637/9e889170-43f5-49ac-aa7f-2760923024d0" width="400" height="300"><br>

### 손실함수
어떤 문제에서 머신러닝 알고리즘이 얼마나 엉터리인지를 측정하는 기준이다. <br>
손실 함수의 값이 작을수록 좋다. <br>

  * 로지스틱 손실 함수 [ 이진 크로스엔트로피 함수 ]<br>
  이진 분류일 경우 로지스틱 손실 함수를 사용한다. <br>
  머신러닝의 경우 정확도로 모델의 성능을 보고 로지스틱 손실 함수를 활용하여 최적화를 한다.<br>

    * 양성클래스 ( 타깃 = 1 ) 일 경우<br>
      -log (예측 확률)로 계산하여 확률이 1에서 멀어질수록 손실은 큰 양수가 된다.
    * 음성클래스 ( 타킷 = 0 ) 일 경우<br>
      -log ( 1 - 예측 확률 )로 계산하여 확률이 0에서 멀어질수록 손실은 큰 양수가 된다.

<img src="https://github.com/piolink/pyree/assets/90364637/895baf6e-6169-4946-8221-f65ba3a134b2" width="300" height="200"><br>

  * 크로스엔트로피 손실 함수<br>
  다중 분류에서 사용하는 함수이다.<br>

  * 평균 제곱 오차<br>
  회귀에서 사용하는 손실함수로, 타깃에서 예측을 뺀 값을 제곱한 다음 모든 샘플에 평균한 값이다.<br>
  값이 작을수록 좋은 모델이다. <br>

### SGDClassifier

아래의 코드는 CH04-1과 동일하다.
```python
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

SGDClassifier는 사이킷런에서 확률적 경사 하강법을 제공하는 대표적인 분류용 클래스이다. <br>
SGDClassifier는 미니배치 경사 하강법과 배치 경사 하강법은 지원하지 않는다.<br>

loss='log'로 지정하여 로지스틱 손실 함수로 지정한다. <br> 
max_iter는 수행할 에크포 횟수를 지정한다. <br>
```python
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```
```
결과값 -> 0.7731과 0.775가 나온다
```

fit()는 가중치와 절편을 모두 버리고 다시 훈련하지만, partial_fit()는 가중치와 절편을 유지하면서 훈련하는 방식이다.<br>
partial_fit()는 이전보다 정확도가 향상된 걸 볼 수 있다.
```python
sc.partial_fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```
```
결과값 -> 0.815와 0.825가 나온다.
```

### 에포크와 과대/과소적합
  적은 에포크 횟수 동안에 훈련한 모델은 훈련 세트와 테스트 세트에 잘 맞지 않는 과소적합된 모델일 가능성이 높다. <br>
  많은 에포크 횟수 동안에 훈련한 모델은 훈련 세트에 너무 잘 맞아 테스트 세트에는 오히려 점수가 나쁜 과대적합된 모델일 가능성이 높다<br><br>
<img src="https://github.com/piolink/pyree/assets/90364637/e5f30266-692b-4eae-82f2-f63ea4c2204d" width="300" height="200"><br>

### 조기종료
  훈련 세트 점수는 에포크가 진행될수록 증가하지만 테스트 세트 점수는 어느 순간 감소한다.<br>
  이 지점이 모델이 과대적합되기 시작하는 곳이고, 과대 적합이 시작되기 전에 훈련을 멈추는 것을 의미한다. <br><br>


  ```python
import numpy as np
sc = SGDClassifier(loss='log_loss', random_state=42)
train_score = []
test_score = []

# classes를 매개변수를 사용하여 partial_fit()는 데이터의 일부분만 전달될 수 있다고 가정하기에 전체 샘플의 클래스 개수를 전달한다.
classes = np.unique(train_target)
# 에포크 횟수를 증가시켜 훈련 세트와 테스트 세트의 차이 나는 걸 볼 수 있다. 
for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

# 300번의 에포크 동안 기록한 훈련 세트와 테스트 세트의 점수를 그래프로 나타낸다.
import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
```
<img src="https://github.com/piolink/pyree/assets/90364637/af017eba-c9fe-41bc-be4f-064d766f2e51" width="300" height="200"><br>

위의 그래프를 통해 에포크가 100일 때 적절한 반복 횟수라는 것을 알 수 있다. <br>
```python
sc = SGDClassifier(loss='log_loss', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```
```
결과값 -> 0.957과 0.925가 나온다.
```

### 힌지손실
힌지손실은 서포트 벡터 머신으로, 머신러닝 알고리즘 위한 손실 함수이다.
```python
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```
