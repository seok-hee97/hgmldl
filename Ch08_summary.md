# CH08 . 이미지를 위한 인공 신경망<br>
## CH08-1 . 합성곱 신경망의 구성 요소

아래의 그림은 7장에서 사용한 밀집층에는 뉴런마다 입력 개수 만큼의 가중치가 있다. <br>
즉, 모든 가중치를 곱해야 한다.<br>

<img src="https://github.com/piolink/pyree/assets/90364637/9de0e976-8e97-4ddc-9cd6-f2cd30e8e431" width="300" height="200"><br><br>
예를 들어, 이미지의 경우 2차원이기 때문에 2차원을 1차원으로 펼치면 효율적으로 처리하지 못한다. <br>
원래의 데이터가 2차원이라면 2차원 데이터로 유지하는 것이 좋다. <br>
이것을 구현해 주는 것이 바로 합성곱 신경망이다.<br><br>

### 합성곱 신경망
* 합성곱은 밀집층과 비슷하게 입력과 가중치를 곱하고 절편을 더하는 선형 계산이다.
* 밀집층과 달리 합성곱은 입력 데이터 전체에 가중치를 적용하지 않고 일부에만 가중치를 곱한다.
* 합성곱 층의 뉴런에 있는 가중치 개수는 정하기 나름이기 때문에 하이퍼파라미터이다.
* 밀칩층의 뉴런을 필터 혹은 커널이라고 부른다. 즉, 뉴런 = 필터 = 커널 


  * 1차원 합성곱<br>
    * 밀집층의 뉴런은 입력 개수만큼 10개의 가중치를 가지고 1개의 출력을 만들지만, 합성곱 층의 뉴런은 3개의 가중치를 가지고 8개의 출력을 만든다.<br>
<img src="https://github.com/piolink/pyree/assets/90364637/25d25af6-0249-4f9a-a3c6-1cbdc1f435cb" width="600" height="300"><br><br>

  * 2차원 합성곱<br>
    * 합성곱의 장점은 2차원 입력에도 적용이 가능하다.
    * 2차원의 경우, 왼쪽에서 오른쪽으로 이동하고 위에서 아래로, 왼쪽에서 오른쪽 이러한 방식으로 이동한다.<br>
<img src="https://github.com/piolink/pyree/assets/90364637/bfa3d04b-e9b5-4415-86e3-42aa4c42f608" width="900" height="300"><br><br>

   * 3차원 이상의 합성곱<br>
  <img src="https://github.com/piolink/pyree/assets/90364637/97c709e2-efac-4867-b962-6e813a2108c1" width="400" height="300"><br><br>

* 합성곱을 계산을 통해 얻은 출력을 특성 맵이라고 부른다.<br>
 * 특성 맵은 활성화 함수를 통과한 값을 나타낸다. <br>
<img src="https://github.com/piolink/pyree/assets/90364637/7ecb4c9c-8603-41ac-8b5c-df359a05f9e1" width="500" height="200"><br><br>


  ### 케라스 합성곱 층
  * 커널의 크기는 하이퍼파라미터이다.
  * 보통 커널의 크기는 (3,3) 혹은 (5,5) 크기를 권장한다.
  ```python
from tensorflow import keras
keras.layers.Conv2D(10, kernel_size=(3, 3), activation='relu')
  ```
### 패딩
* 입력 배열의 주위를 가상의 원소로 채우는 것을 의미한다.<br>
   * 세임 패딩은 입력과 특성 맵의 크기를 동일하게 만들기 위해 입력 주위에 0으로 패딩하는 것을 의미한다.
   * 합성곱 신경망에서는 세임 패딩이 많이 사용된다.<br>
  <img src="https://github.com/piolink/pyree/assets/90364637/ba155706-8566-4265-8292-fae041753678" width="200" height="200"><br><br>
   * 밸리드 패딩은 패딩 없이 순수한 입력 배열에서만 합성곱을 하여 특성 맵을 만드는 경우이다.
   * 특성 맵의 크기가 줄어들 수 밖에 없다.<br>
   <img src="https://github.com/piolink/pyree/assets/90364637/6fdc8948-f215-42fc-933c-ff8aa018d9ef" width="200" height="200"><br><br>

### 패딩의 목적
* 패딩을 하지 않을 경우 중앙부와 모서리 팍셀이 합성곱에 참여하는 비율은 크게 차이난다.<br>
* 패딩은 이미지의 주변에 있는 정보를 잃어버리지 않도록 도와준다.<br>
 <img src="https://github.com/piolink/pyree/assets/90364637/1fadd610-3c59-4dc3-a4e3-01614c515311" width="300" height="200"><br><br>
 ```python
# 케라스의 패딩 설정
keras.layers.Conv2D(10, kernel_size=(3,3), activation='relu', padding='same')
```

### 스프라이드
* 합성곱 층에서 필터가 입력 위를 이동하는 크기로 기본으로 스프라이트는 1픽셀, 즉 1칸씩 이동한다.<br>
* strides = 2로 설정할 경우, 밸리드 패딩 효과를 볼 수 있다.<br>
  <img src="https://github.com/piolink/pyree/assets/90364637/4298bec6-fea3-4c77-bc62-c5339f8b117b" width="400" height="200"><br><br>
  ```python
  # 케라스의 패딩 설정
  keras.layers.Conv2D(10, kernel_size=(3,3), activation='relu', padding='same', strides=1)
  ```

### 폴링
* 합성곱 층에서 만든 특성 맵의 가로세로 크기를 줄이는 역할을 한다.
* 하지만 특성 맵의 크기는 줄이지 않는다.<br>
<img src="https://github.com/piolink/pyree/assets/90364637/c44ca331-e008-42e0-8d9e-06526fe71c66" width="400" height="200"><br><br>
  * 도장을 찍은 영역에서 가장 큰 값을 고르거나 퍙균값을 계산한 값을 각각 최대 폴링과 평균 폴링이라고 한다.<br>
  <img src="https://github.com/piolink/pyree/assets/90364637/b2ac48b8-28d5-4d1b-b5be-0a7ac6fa0a78" width="400" height="200"><br><br>
```python
# 케라스 풀링 
keras.layers.MaxPooling2D(2)
keras.layers.Maxpooling2D(2, strides=2, padding='valid')
```
### 합성곱 신경망의 전체 구조
<img src="https://github.com/piolink/pyree/assets/90364637/f9e333e1-041e-45a4-b096-737d8c7ce9a2" width="400" height="200"><br>
1. 4x4 입력을 패딩을 추가하여 6x6 입력으로 변경한다.<br>
2. 필터 3개를 통해 특성 맵을 적용한다. 특성 맵은 4x4 크기로 된다.<br>
3. 활성화 함수가 적용되어 최종 특성 맵이 생긴다,<br>
4. 최대 풀링을 적용하여 2x2 특성 맵이 생긴다.<br>
5. 7장에서 나왔듯이 특성 먑(2,2,3)이 flatten()을 통해 1차원 배열의 뉴런이 생긴다.<br>
6. 은닉층 혹은 입력층을 통과하여 예측한다.
   
<br>
<br>

## CH08-2 . 합성곱 신경망을 사용한 이미지 분류
<br>

### 패션 MNIST 데이터
```python
from tensorflow import keras
from sklearn.model_selection import train_test_split
# keras.datasets.fashion_mnist 모듈 아래 load_data() 함수를 통해 데이터를 반환
(train_input, train_target), (test_input, test_target) =\
    keras.datasets.fashion_mnist.load_data()
# 255로 나누어 0 ~ 1로 정규화
train_scaled = train_input.reshape(-1,28,28,1) / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
```

### 합성곱 신경망 만들기

#### 첫 번째 합성곱 층
```python
# sequential 클래스의 객체를 만든다.
model = keras.Sequential()
# 합성곱 층인 Conv2D를 추가하여 모델의 add() 메서드를 사용해 층을 하나씩 추가한다. 32개의 필터 사용, 3 x 3 커널의 크기 
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1)))
# 최대 풀링인 경우 -> MaxPooling2D(2), 평균 풀링인 경우 -> AveragePooling2D(2) 클래스 제공
model.add(keras.layers.MaxPooling2D(2))

```
 <img src="https://github.com/piolink/pyree/assets/90364637/80181bdb-404a-4fca-9071-39ca0ebd3ef1" width="500" height="200"><br><br>


#### 두 번째 합성곱 층 + 완전 연결층
```python
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
# dropout은 은닉층과 출력층 사이에 과대적합을 막아 성능을 개선해준다.
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))
```
<img src="https://github.com/piolink/pyree/assets/90364637/9706d426-d7fd-479a-8726-c09c5c7a32a5" width="800" height="300"><br><br>

#### 모델 구성
```pyhton
model.summary()
```
<img src="https://github.com/piolink/pyree/assets/90364637/13bad42d-bdb7-45b3-8b4e-bab8fde7d39e" width="500" height="600"><br><br>

<br>
케라스는 summary()외에 층의 구성을 그림으로 표현해주는 plot_model() 함수가 존재한다.<br>

```python
keras.utils.plot_model(model, show_shapes=True)
```
<img src="https://github.com/piolink/pyree/assets/90364637/e0e8322b-eeb6-4bad-a1c3-22ade6e070c9" width="500" height="600"><br><br>

#### 모델 컴파일과 훈련
```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])
```

```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

```python
model.evaluate(val_scaled, val_target)
```

```python
plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
plt.show()
```


### CH08-3 . 합성곱 신경망의 시각화

#### 가중치 시각화
* 가중치는 입력 이미지의 2차원 영역에 적용되어 어떤 특징을 크게 두드러지게 표현하는 역할을 한다.
* 아래의 그림처럼 이 필터의 가운데 곡선 부분의 가중치 값은 높고 그 외의 가중치 값은 낮을 것이다.
   * 둥근 모서리가 있는 입력과 곱해져서 큰 출력을 만든다.<br>
  <img src="https://github.com/piolink/pyree/assets/90364637/f66d25ca-0e85-4c74-b4e3-9a7a4fef0482" width="300" height="200"><br><br>

```python
from tensorflow import keras
# best-cnn-model.h5는 ch08-2에서 파일을 생성한 다음 이어서 해야 한다.
model = keras.models.load_model('best-cnn-model.h5')
```

케라스 모델에 추가한 층은 layers에 저장되어 있다.<br>
이 속성은 파이썬 리스트이다.<br>
```python
model.layers
```
  <img src="https://github.com/piolink/pyree/assets/90364637/e9135461-ca48-4c1f-9ee2-38c796ddc14d" width="600" height="300"><br><br>
