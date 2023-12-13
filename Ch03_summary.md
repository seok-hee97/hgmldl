
# Chapter03 회귀 알고리즘과 모델 규제
## 농어의 무게를 예측하라!
<br>
<br>

## 03-1 k-최근접 이웃 회귀
- 예측하려는 샘플 x의 타깃을 예측하는 간단한 방법은 수치들의 평균을 구하면 됨

  <img src="https://github.com/piolink/pyree/assets/67042526/50138dba-0067-4482-8cfd-f829175f2cc1"  width="300">
  <br><br>
  
### 데이터 준비
  - 농어 데이터(넘파이 배열)
    
    ```python
    import numpy as np
    
    perch_length = np.array(
        [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 
         21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 
         22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 
         27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 
         36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 
         40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
         )
    perch_weight = np.array(
        [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
         110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
         130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
         197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
         514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
         820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
         1000.0, 1000.0]
         )
    ```
    
    - 산점도(특성:x축, 타깃:y축)
        ```python
        import matplotlib.pyplot as plt
        
        plt.scatter(perch_length, perch_weight)
        plt.xlabel('length')
        plt.ylabel('weight')
        plt.show()
        ```
        <img src="https://github.com/piolink/pyree/assets/67042526/a45de4d2-c8f3-4fd7-9d4c-61aa4e5e1b56"  width="400">
<br>

  - 농어 데이터를 훈련 세트와 테스트 세트로 나누기
    
    ```python
    from sklearn.model_selection import train_test_split
    
    train_input, test_input, train_target, test_target = train_test_split(
        perch_length, perch_weight, random_state=42)
    
    print(train_input.shape, test_input.shape)
    ```
  <br>
  
  - 사이킷런에 사용할 훈련 세트는 2차원 배열이어야하기 때문에 train_input, test_input을 아래와 같이 1차원 배열에서 2차원 배열로 수정

    <img src="https://github.com/piolink/pyree/assets/67042526/cc69fccd-f3c2-4cdd-a55e-555cb9d4e5e8" width="200">
  <br>
  
  - 넘파이 배열에서 크기를 바꿀 수 있는 reshape() 메서드
    - reshape() 메서드 : 크기가 바뀐 새로운 배열을 반환할 때 지정한 크기가 원본 배열에 있는 원소의 개수와 다르면 에러가 발생   
      ex) (4, ) → (2, 3)은 안됨
      
      ```python
      #예시
      test_array = np.array([1,2,3,4])
      print(test_array.shape)
      
      test_array = test_array.reshape(2, 2)
      print(test_array.shape)
      
      # 아래 코드의 주석을 제거하고 실행하면 에러가 발생합니다
      # test_array = test_array.reshape(2, 3)
      ```
  <br>
  
  - train_input, test_input 2차원 배열로 수정
    - 크기에 -1을 지정하면 나머지 원소 개수로 모두 채움

       ```python
        train_input = train_input.reshape(-1, 1)
        test_input = test_input.reshape(-1, 1)
        print(train_input.shape, test_input.shape)
       ```
<br><br><br>

### 결정계수_R<sup>2<sup/>

- 회귀 모델 훈련
    
    ```python
    from sklearn.neighbors import KNeighborsRegressor
    
    knr = KNeighborsRegressor()
    
    # k-최근접 이웃 회귀 모델을 훈련합니다
    knr.fit(train_input, train_target)
    ```
  <br>
  
- 테스트 세트 점수 확인
    
    ```python
    print(knr.score(test_input, test_target))
    ```
  <br>

- 회귀에서는 정확한 숫자를 맞힌다는 것은 거의 불가능함 → 예측하는 값이나 타깃 모두 임의의 수치이기 때문
- 회귀의 경우 다른 값으로 평가하는데 이 점수를 **결정계수, R<sup>2<sup/>**
    
    <img src="https://github.com/piolink/pyree/assets/67042526/f8afc8a5-93ae-4698-8477-f4158acd1bd5" width="200">
    
    - 타깃의 평균 정도를 예측하는 수준이라면 R^2는 0에 가까워지고, 예측이 타깃에 아주 가까워지면 1에 가까운 값이 됨
<br>

- 타깃과 예측의 절댓값 오차를 평균하여 반환하는 mean_absolute_error
    
    ```python
    from sklearn.metrics import mean_absolute_error
    
    # 테스트 세트에 대한 예측을 만듭니다
    test_prediction = knr.predict(test_input)
    
    # 테스트 세트에 대한 평균 절댓값 오차를 계산합니다
    mae = mean_absolute_error(test_target, test_prediction)
    print(mae)
    ```
  <br><br><br>

### 과대적합 vs 과소적합
- 훈련한 모델을 사용하여 훈련 세트의 R^2 점수를 확인
    
    ```python
    print(knr.score(train_input, train_target))
    ```
  <br>

- 훈련 세트에서 점수가 좋았는데 테스트 세트에서 점수가 나쁘다면 모델이 훈련 세트에 **과대적합**이 됨
- 훈련 세트보다 테스트 세트의 점수가 높거나 두 점수가 모두 너무 낮은 경우는 모델이 훈련 세트에 **과소적합**이 됨
- 과소적합의 문제를 해결하는 방법은 모델을 조금 더 복잡하게 만들면 됨
    - k-최근접 이웃 알고리즘으로 모델을 더 복잡하게 만드는 방법은 이웃의 개수 k를 줄이면 됨
<br>

- 이웃의 개수를 줄임
    
    ```python
    # 이웃의 갯수를 3으로 설정합니다
    knr.n_neighbors = 3
    
    # 모델을 다시 훈련합니다
    knr.fit(train_input, train_target)
    print(knr.score(train_input, train_target))
    
    #테스트 세트 점수 확인
    print(knr.score(test_input, test_target))
    ```
<br><br><br><br>    

### 최종 코드

3-1.ipynb : [github](https://github.com/rickiepark/hg-mldl/blob/master/3-1.ipynb)
<br><br><br>

### 핵심 패키지와 함수
- scikit-learn
    - KNeighborsRegressor
    - mean_absolute_error()
- numpy
    - reshape()
<br><br><br>

### 확인 문제
- 과대적합과 과소적합에 대한 이해를 돕기 위해 복잡한 모델과 단순한 모델을 만듬,   
  앞서 만든 k-최근접 이웃 회귀 모델의 k 값을 1, 5, 10으로 바꾸며 훈련,   
  그다음 농어의 길이를 5에서 45까지 바꿔가며 예측을 만들어 그래프로 나타내기
    
    ```python
    # k-최근접 이웃 회귀 객체를 만듭니다
    knr = KNeighborsRegressor()
    
    # 5에서 45까지 x 좌표를 만듭니다
    x = np.arange(5, 45).reshape(-1, 1)
    
    # n = 1, 5, 10일 때 예측 결과를 그래프로 그립니다.
    for n in [1, 5, 10]:
        # 모델 훈련
        knr.n_neighbors = n
        knr.fit(train_input, train_target)
    
        # 지정한 범위 x에 대한 예측 구하기 
        prediction = knr.predict(x)
    
        # 훈련 세트와 예측 결과 그래프 그리기
        plt.scatter(train_input, train_target)
        plt.plot(x, prediction)
        plt.title('n_neighbors = {}'.format(n))    
        plt.xlabel('length')
        plt.ylabel('weight')
        plt.show()
    ```
    
    <img src="https://github.com/piolink/pyree/assets/67042526/d1a9898e-1f41-4728-a045-cc207d3dad78" width="400">
    
    <img src="https://github.com/piolink/pyree/assets/67042526/96c9ad16-dbd1-49ed-bb4f-9101613d6dbd" width="400">
    
    <img src="https://github.com/piolink/pyree/assets/67042526/022e4c75-f545-49e6-9840-7f4474286f82" width="400">
    
<br><br><br><br><br><br><br>

## 03-2 선형 회귀

### k-최근접 이웃의 한계

- 데이터와 모델 준비
  
  ```python
    import numpy as np
    
    perch_length = np.array(
        [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 
         21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 
         22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 
         27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 
         36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 
         40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
         )
    perch_weight = np.array(
        [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
         110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
         130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
         197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
         514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
         820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
         1000.0, 1000.0]
         )
    ```
  <br>

- 훈련 세트와 데스트 세트로 분류 (특성 데이터는 2차원 배열로 변환)
  
  ```python
    from sklearn.model_selection import train_test_split
        
    # 훈련 세트와 테스트 세트로 나눕니다
    train_input, test_input, train_target, test_target = train_test_split(
        perch_length, perch_weight, random_state=42)
    
    # 훈련 세트와 테스트 세트를 2차원 배열로 바꿉니다
    train_input = train_input.reshape(-1, 1)
    test_input = test_input.reshape(-1, 1)
  ```
<br>

- 최근접 이웃 개수를 3으로 하는 모델 훈련

  ```python
    from sklearn.neighbors import KNeighborsRegressor
        
    knr = KNeighborsRegressor(n_neighbors=3)
    
    # k-최근접 이웃 회귀 모델을 훈련합니다
    knr.fit(train_input, train_target)
  ```
<br>

- 길이가 50cm인 농어의 무게 예측

    ```python
      print(knr.predict([[50]]))
    ```
<br>

- 훈련 세트, 50cm 농어, 농어의 최근접 이웃을 산점도로 표시

    ```python
      import matplotlib.pyplot as plt
  
      # 50cm 농어의 이웃을 구합니다
      distances, indexes = knr.kneighbors([[50]])
      
      # 훈련 세트의 산점도를 그립니다
      plt.scatter(train_input, train_target)
  
      # 훈련 세트 중에서 이웃 샘플만 다시 그립니다
      plt.scatter(train_input[indexes], train_target[indexes], marker='D')
  
      # 50cm 농어 데이터
      plt.scatter(50, 1033, marker='^')
      plt.xlabel('length')
      plt.ylabel('weight')
      plt.show()
    ```

  <img src="https://github.com/piolink/pyree/assets/67042526/1f25c43c-8d86-4946-9933-3b119881ebfd" width="400">

  - 이웃 샘플의 타깃의 평균

      ```python
        print(np.mean(train_target[indexes]))
      ```
      - k-최근접 이웃 회귀는 가장 가까운 샘플을 찾아 타깃을 평균하기 때문에 새로운 샘플이 훈련 세트의 범위를 벗어나면 엉뚱한 값을 예측할 수 있음
<br>

  - (예시) 길이가 100cm인 농어의 무게 예측

    ```python
      print(knr.predict([[100]]))
    ```
    
    - 그래프로 확인

      ```python
        import matplotlib.pyplot as plt
      
        # 100cm 농어의 이웃을 구합니다
        distances, indexes = knr.kneighbors([[100]])
        
        # 훈련 세트의 산점도를 그립니다
        plt.scatter(train_input, train_target)
      
        # 훈련 세트 중에서 이웃 샘플만 다시 그립니다
        plt.scatter(train_input[indexes], train_target[indexes], marker='D')
      
        # 100cm 농어 데이터
        plt.scatter(100, 1033, marker='^')
        plt.xlabel('length')
        plt.ylabel('weight')
        plt.show()
      ```

      <img src="https://github.com/piolink/pyree/assets/67042526/79a94110-217c-406c-b484-0912a4d1e04f" width="400">
      
      - 농어가 아무리 커도 무게가 더 늘어나지 않음
    <br>

- 머신러닝 모델은 주기적으로 훈련해야 함
  - 시간과 환경이 변화하면서 데이터도 바뀌기 때문에 주기적으로 새로운 훈련 데이터로 모델을 다시 훈련해야 함  
<br><br><br>


### 선형 회귀

- 널리 사용되는 대표적인 회귀 알고리즘으로 비교적 간단하고 성능이 뛰어남
- 특성이 하나인 경우 어떤 직선(특성을 가장 잘 나타낼 수 있는 직선)을 학습하는 알고리즘
  - 예시
    
    <img src="https://github.com/piolink/pyree/assets/67042526/941ecf60-a93f-4b32-ac0d-881fba5f9680" width="600">
    
    - (1)R^2=0 / (2)R^2=음수 / (3)유사
  <br>

- 사이킷런은 sklearn.linear_model 패키지 아래 LinearRegression 클래스로 선형 회귀 알고리즘 구현
  - 사이킷런 모델 클래스들은 훈련[fit()], 평가[score()], 예측[predict()]하는 메서드 이름이 모두 동일

    ```python
      from sklearn.linear_model import LinearRegression
      lr = LinearRegression()
    
      # 선형 회귀 모델 훈련
      lr.fit(train_input, train_target)
      
      # 50cm 농어에 대한 예측
      print(lr.predict([[50]]))
    ```

    - k-최근접 이웃 회귀보다 선형 회귀가 무게를 더 높게 예측
    - 선형 회귀가 학습한 직선은 **y = a*x+b** 직선의 방정식 사용 (x:농어의 길이, y:농어의 무게)
  
       <img src="https://github.com/piolink/pyree/assets/67042526/be72c000-6b89-408a-a59b-b205c51168b7" width="350">
    <br>

    - LinearRegression 클래스가 찾은 a와 b는 lr 객체의 coef_와 intercept_ 속성에 저장되어 있음

      ```python
        print(lr.coef_, lr.intercept_)
      ```
    <br>
  
- coef_와 intercept_를 머신러닝 알고리즘이 찾은 값이라는 의미로 **모델 파라미터**라고 함
- 많은 머신러닝 알고리즘의 훈련 과정은 최적의 모델 파라미터를 찾는 것과 같으며 **모델 기반 학습**이라고 함
- 모델 파라미터가 없는 k-최근접 이웃과 같은 알고리즘의 훈련 과정을 **사례 기반 학습**이라고 함
<br>

- 농어의 길이 15 ~ 50까지를 직선으로 그리며 훈련 세트의 산점도와 함께 생성

  ```python
    import matplotlib.pyplot as plt
  
    # 훈련 세트의 산점도를 그립니다
    plt.scatter(train_input, train_target)
  
    # 15에서 50까지 1차 방정식 그래프를 그립니다
    plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])
  
    # 50cm 농어 데이터
    plt.scatter(50, 1241.8, marker='^')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
  ```
  
  <img src="https://github.com/piolink/pyree/assets/67042526/70351229-f692-4c10-805e-86742145fe7b" width="400">
  
  - 선형 회귀 알고리즘이 이 데이터셋에서 찾은 최적의 직선
  - 길이가 50cm인 농어에 대한 예측이 직선의 연장선에 있기 때문에 훈련 세트 범위 밖의 무게도 예측 가능
<br>

  - 훈련 세트와 테스트 세트에 대한 R^2 점수 확인
 
    ```python
      print(lr.score(train_input, train_target))  #훈련 세트
      print(lr.score(test_input, test_target))   #테스트 세트
    ```

    - 훈련 세트와 테스트 세트의 점수가 조금 차이가 나며 훈련 세트의 점수도 높지 않음(과소적합) + 그래프 왼쪽 아래 값 이상함
<br><br><br>


### 다항 회귀

- 위에서 그린 선형 회귀가 만든 직선은 왼쪽 아래로 쭉 뻗어있어 직선대로 예측하면 농어의 무게가 0g 이하로 내려감
- 최적의 직선을 찾기보다는 최적의 곡선을 찾는게 더 좋은 방법

  <img src="https://github.com/piolink/pyree/assets/67042526/48a46890-6867-45fc-bf9a-2ffaaf1b0352" width="350">  
  
  - 넘파이를 사용하면 위와 같은 2차 방정식의 그래프를 간단히 만들 수 있음
<br>

- 농어의 길이를 제곱하여 원래 데이터 옆에 붙이기
  - 제곱한 식에도 넘파이 브로드캐스팅 적용됨  

    <img src="https://github.com/piolink/pyree/assets/67042526/1851d08f-0227-49c4-8283-f3161158d6fe" width="180">  

    ```python
      train_poly = np.column_stack((train_input ** 2, train_input))
      test_poly = np.column_stack((test_input ** 2, test_input))
  
      #새롭게 만든 데이터셋의 크기 확인
      print(train_poly.shape, test_poly.shape)
    ```  
  <br>

- train_poly를 사용해 선형 회귀 모델을 훈련
  - 2차 방정식 그래프를 찾기 위해 훈련 세트에 제곱 항을 추가했지만, 타깃값은 그대로 사용   
      -> 목표하는 값은 어떤 그래프를 훈련하던지 바꿀 필요가 없음  

    ```python
      lr = LinearRegression()
      lr.fit(train_poly, train_target)
      
      print(lr.predict([[50**2, 50]]))
    ```  
<br>

  - 훈련한 계수와 절편 출력
    
    ```python
      print(lr.coef_, lr.intercept_)
    ```

    - 모델이 학습한 그래프

      ```
        무게 = 1.01 x 길이^2 - 21.6 x 길이 + 116.05
      ```
    <br>

- 이런 방정식을 다항식이라 하며 다항식을 사용한 선형 회귀를 **다항회귀**라고 부름
<br>

- 훈련 세트의 산점도에 그래프로 생성
  - 짧은 직선을 이어서 그리면 곡선처럼 표현 가능

    ```python
      import matplotlib.pyplot as plt
    
      # 구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만듭니다
      point = np.arange(15, 50)
    
      # 훈련 세트의 산점도를 그립니다
      plt.scatter(train_input, train_target)
    
      # 15에서 49까지 2차 방정식 그래프를 그립니다
      plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)
    
      # 50cm 농어 데이터
      plt.scatter([50], [1574], marker='^')
      plt.xlabel('length')
      plt.ylabel('weight')
      plt.show()
    ```

    <img src="https://github.com/piolink/pyree/assets/67042526/2d8052f2-556e-4a4f-8095-96a39b4a02bf" width="400"> 
  <br>

    - 훈련 세트와 테스트 세트의 R^2 점수 평가

      ```python
        print(lr.score(train_poly, train_target))
        print(lr.score(test_poly, test_target))
      ```

      - 두 개의 점수 모두 크게 높아졌지만 테스트 세트의 점수가 여전히 조금 더 높음   
        -> 과소적합이 남아 있음 
    <br><br><br><br>    


### 최종 코드

3-2.ipynb : [github](https://github.com/rickiepark/hg-mldl/blob/master/3-2.ipynb)
<br><br><br>

### 핵심 패키지와 함수
- scikit-learn
  - LinearRegression
    
<br><br><br><br><br><br><br>


## 03-3 특성 공학과 규제

### 다중 회귀

- 여러 개의 특성을 사용한 선형 회귀를 **다중 회귀**라고 함
- 특성이 2개면 선형 회귀는 평면을 학습함
   
  <img src="https://github.com/piolink/pyree/assets/67042526/4b98a752-5563-4aed-9981-957853159475" width="450">

  - 오른쪽 그림처럼 특성이 2개면 타깃값과 함께 3차원 공간을 형성   
    -> 선형 회귀 방정식 *'타깃 = a x 특성1 + b x 특성2 + 절편'* 은 평면이 됨 
<br>

- 3차원 공간 이상은 그릴 수 없지만 특성이 많은 고차원에서는 선형 회귀가 매우 복잡한 모델을 표현할 수 있음
- 기존의 특성을 사용해 새로운 특성을 뽑아내는 작업을 **특성 공학**이라고 함
<br><br><br>

### 데이터 준비

- **판다스**는 유명한 데이터 분석 라이브러리
- **데이터프레임**은 판다스의 핵심 데이터 구조
  - 넘파이 배열과 비슷하게 다차원 배열을 다룰 수 있지만 훨씬 더 많은 기능을 제공, 넘파이 배열로 쉽게 변경 가능
  - 판다스 데이터프레임을 만들기 위해 많이 사용하는 파일은 CSV 파일(콤마로 나누어져 있는 텍스트 파일)

    <img src="https://github.com/piolink/pyree/assets/67042526/8ccfd7e9-8d0d-4472-a34a-4ee282fcf9f8" width="220">

    - 전체 파일 내용: [perch_csv_data](https://bit.ly/perch_csv_data)
 
    - 판다스를 사용해 농어 데이터를 데이터프레임에 저장 후 넘파이 배열로 변환하여 선형 회귀 모델 훈련
      - 파일을 판다스에서 읽는 방법
   
        ```python
          import pandas as pd  #pd는 관례적으로 사용하는 판다스의 별칭
        
          df = pd.read_csv('https://bit.ly/perch_csv_data')
          perch_full = df.to_numpy()
          print(perch_full)
        ```
        
        <img src="https://github.com/piolink/pyree/assets/67042526/54127dc1-0e5e-4620-88e0-cbafe2395afa" width="500">
    <br>

- 타깃 데이터

  ```python
    import numpy as np

    perch_weight = np.array(
        [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
         110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
         130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
         197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
         514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
         820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
         1000.0, 1000.0]
         )
  ```  
<br>

- perch_full과 perch_weight를 훈련 세트와 테스트 세트로 분리

  ```python
    from sklearn.model_selection import train_test_split

    train_input, test_input, train_target, test_target = train_test_split(
        perch_full, perch_weight, random_state=42)
  ```
<br><br><br>

### 사이킷런의 변환기

- 사이킷런은 특성을 만들거나 전처리하기 위한 다양한 클래스를 제공   
  - 이러한 클래스를 **변환기**라고 함   
    - 모델 클래스(추정기) : fit(), score(), predict() 메서드 제공   
    - 변환기 클래스 : fit()[훈련], transform()[변환] 메서드 제공   
        +) 두 메서드를 하나로 붙인 fit_transform() 메서드
<br>

- (사용 방법) PolynomialFeatures 클래스 변환기 사용

  ```python
    from sklearn.preprocessing import PolynomialFeatures

    #2개의 특성 2와 3으로 이루어진 샘플 적용
    poly = PolynomialFeatures()
    poly.fit([[2, 3]])
    print(poly.transform([[2, 3]]))
  ```

  - 변환기는 입력 데이터를 변환하는데 타깃 데이터가 필요하지 않음   
    -> 모델 클래스와는 다르게 fit() 메서드에 입력 데이터만 전달
  <br>

    - PolynomialFeatures 클래스는 기본적으로 각 특성을 제곱한 항을 추가하고 특성끼리 서로 곱한 항을 추가
      - 선형 방정식의 절편을 항상 값이 1인 특성과 곱해지는 계수
      - 특성은 (길이, 높이, 두께, 1)
      
        ```
          무게 = a x 길이 + b x 높이 + c x 두께 + d x 1
        ```
    <br>
    
    - 사이킷런의 선형 모델은 자동으로 절편을 추가하므로 include_bias=False로 지정하여 특성을 변환
  
      ```python
        poly = PolynomialFeatures(include_bias=False)
        poly.fit([[2, 3]])
        print(poly.transform([[2, 3]]))
      ```
  
      - 절편을 위한 항이 제거, 특성의 제곱과 특성끼리 곱한 항만 추가
      - 굳이 include_bias=False로 지정하지 않아도 사이킷런 모델은 자동으로 특성에 추가된 절편 항을 무시
  <br><br>

- 위 방식으로 train_input에 적용

   ```python
     poly = PolynomialFeatures(include_bias=False)
     poly.fit(train_input)
     train_poly = poly.transform(train_input)  #train_input을 변환한 데이터를 저장
   
     print(train_poly.shape)  #배열의 크기 확인
   ```
<br>

- 9개의 특성이 각각 어떤 입력의 조합으로 만들어졌는지 알려주는 get_feature_names_out() 메서드

  ```python
    poly.get_feature_names_out()
  ```

  - 'x0' : 첫 번째 특성
  - 'x0^2' : 첫 번째 특성의 제곱
  - 'x0 x1' : 첫 번째 특성과 두 번째 특성의 곱
<br>

- 테스트 세트 변환

  ```python
    test_poly = poly.transform(test_input)
  ```  
<br><br><br>


### 다중 회귀 모델 훈련하기

- 다중 회귀 모델을 훈련하는 것은 선형 회귀 모델을 훈련하는 것과 같음, 다만 여러 개의 특성을 사용하여 선형 회귀를 수행
<br>

- train_poly를 사용해 모델을 훈련, 점수 확인

  ```python
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(train_poly, train_target)
  
    print(lr.score(train_poly, train_target))
  ```
<br>

- 테스트 세트에 대한 점수 확인

  ```python
    print(lr.score(test_poly, test_target))
  ```

- 농어의 길이, 높이, 두께를 모두 사용하고 각 특성을 제곱하거나 서로 곱해서 다항 특성을 더 추가   
  -> 테스트 세트에 대한 점수는 높아지지 않았지만 농어의 길이만 사용했을 때 발생한 과소적합 문제가 사라짐
<br>

- PolynomialFeatures 클래스의 degree 매개변수(필요한 고차항의 최대 차수 지정)를 사용해 5제곱까지 특성을 만들어 출력

  ```python
    poly = PolynomialFeatures(degree=5, include_bias=False)
    poly.fit(train_input)
    train_poly = poly.transform(train_input)
    test_poly = poly.transform(test_input)
  
    print(train_poly.shape)
  ```

  - 위 데이터를 사용해 선형 회귀 모델 훈련

    ```python
      lr.fit(train_poly, train_target)
      print(lr.score(train_poly, train_target))
    ```

  - 테스트 세트 점수

    ```python
      print(lr.score(test_poly, test_target))
    ```

    - 훈련 세트는 거의 완벽한 점수가 나오지만 테스트 세트는 음수가 나옴   
      -> 특성의 개수를 크게 늘리면 선형 모델은 강력해지지만 훈련 세트에 너무 과대적합되므로 테스트 세트에서는 이상한 점수를 만듬
<br><br><br>


### 규제

- 머신러닝 모델이 훈련 세트를 너무 과도하게 학습하지 못하도록 훼방하는 것
- 즉 모델이 훈련 세트에 과대적합되지 않도록 만드는 것
- 선형 회귀 모델의 경우 특성에 곱해지는 계수(기울기)의 크기를 작게 만드는 일

  <img src="https://github.com/piolink/pyree/assets/67042526/c72dd239-5eec-4297-a7b5-950dd6a1ee82" width="500">
<br>

- 앞에서 훈련한 모델의 계수를 규제하여 훈련 세트의 점수를 낮추고 테스트 세트의 점수를 높이기
  - 특성의 스케일이 정규화되지 않으면 곱해지는 계수 값도 차이가 나게 됨 -> 규제를 적용하기 전 정규화
  <br>

    - StandardScaler 클래스 사용
  
      ```python
        from sklearn.preprocessing import StandardScaler
        
        ss = StandardScaler()
        ss.fit(train_poly)
        
        train_scaled = ss.transform(train_poly)
        test_scaled = ss.transform(test_poly)
      ```
  
      - 꼭 훈련 세트로 학습한 변환기를 사용해 테스트 세트까지 변환해야 함
      - 훈련 세트에서 학습한 평균과 표준편차는 StandardScaler 클래스 객체의 mean_, scale_ 속성에 저장
  <br>

- 선형 회귀 모델에 규제를 추가한 모델을 **릿지**와 **라쏘**라고 함
- 일반적으로 릿지를 조금 더 선호
<br><br><br>


### 릿지 회귀

- 릿지와 라쏘 모두 sklearn.linear_model 패키지 안에 있음
<br>

- train_scaled 데이터로 릿지 모델을 훈련

  ```python
    from sklearn.linear_model import Ridge
    
    ridge = Ridge()
    ridge.fit(train_scaled, train_target)
    print(ridge.score(train_scaled, train_target))
  ```
  
  - 테스트 세트 점수
  
     ```python
       print(ridge.score(test_scaled, test_target))
     ```
  
  -> 훈련 세트에 너무 과대적합되지 않아 테스트 세트에서도 좋은 성능을 냄  
  <br>

- 릿지와 라쏘 모델을 사용할 때 규제의 양을 임의로 조절 가능 -> alpha 매개변수   
- alpha 값은 사전에 직접 지정 해줘야되는 값   
  - 머신러닝 모델이 학습할 수 없고 사람이 알려줘야 하는 파라미터를 **하이퍼파라미터**라고 함   
    - 머신러닝 라이브러리에서 하이퍼파라미터는 클래스와 메서드의 매개변수로 표현됨
<br>

- 적절한 alpha값을 찾는 한 가지 방법은 alpha 값에 대한 R^2값의 그래프 그리기
- 훈련 세트와 테스트 세트의 점수가 가장 가까운 지점이 최적의 alpha 값

  ```python
    import matplotlib.pyplot as plt
    
    train_score = []
    test_score = []

    #alpha값을 0.001에서 100까지 10배씩 늘려가며 릿지 회귀 모델을 훈련한 다음
    # 훈련 세트와 테스트 세트의 점수를 리스트에 저장
    alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
    for alpha in alpha_list:
        # 릿지 모델을 만듭니다
        ridge = Ridge(alpha=alpha)
        # 릿지 모델을 훈련합니다
        ridge.fit(train_scaled, train_target)
        # 훈련 점수와 테스트 점수를 저장합니다
        train_score.append(ridge.score(train_scaled, train_target))
        test_score.append(ridge.score(test_scaled, test_target))

    #alpha_list에 있는 6개의 값을 동일한 간격으로 나타내기 위해 로그 함수로 바꾸어 지수로 표현
    plt.plot(np.log10(alpha_list), train_score)
    plt.plot(np.log10(alpha_list), test_score)
    plt.xlabel('alpha')
    plt.ylabel('R^2')
    plt.show()
  ```

  <img src="https://github.com/piolink/pyree/assets/67042526/f3514018-b212-4717-b984-97d3d945036b" width="400">

    - 적절한 alpha 값은 두 그래프가 가장 가깝고 테스트 세트 점수가 가장 높은 -1, 10^-1=0.1   
<br>

  - alpha : 0.1로 하여 최종 모델 훈련

    ```python
      ridge = Ridge(alpha=0.1)
      ridge.fit(train_scaled, train_target)
      
      print(ridge.score(train_scaled, train_target))
      print(ridge.score(test_scaled, test_target))
    ```
<br><br><br>


### 라쏘 회귀

- 릿지와 매우 비슷하며 Ridge 클래스를 Lasso 클래스로 변경

  ```python
    from sklearn.linear_model import Lasso
    
    lasso = Lasso()
    lasso.fit(train_scaled, train_target)
  
    print(lasso.score(train_scaled, train_target))

    #테스트 세트 점수
    print(lasso.score(test_scaled, test_target))
  ```
<br>

- alpha 매개변수로 규제의 강도 조절
  
  ```python
    train_score = []
    test_score = []
    
    alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
    for alpha in alpha_list:
        # 라쏘 모델을 만듭니다
        lasso = Lasso(alpha=alpha, max_iter=10000)
        # 라쏘 모델을 훈련합니다
        lasso.fit(train_scaled, train_target)
        # 훈련 점수와 테스트 점수를 저장합니다
        train_score.append(lasso.score(train_scaled, train_target))
        test_score.append(lasso.score(test_scaled, test_target))
  ```
<br>

- train_score, test_score 리스트를 사용해 그래프 생성(x축은 로그 스케일로 변경)

  ```python
    plt.plot(np.log10(alpha_list), train_score)
    plt.plot(np.log10(alpha_list), test_score)
    plt.xlabel('alpha')
    plt.ylabel('R^2')
    plt.show()
  ```

  <img src="https://github.com/piolink/pyree/assets/67042526/ae036220-337b-47bf-ba70-fe614715ee06" width="400">
<br>

- alpha : 1, 10^1=**10**값으로 모델 훈련

  ```python
    lasso = Lasso(alpha=10)
    lasso.fit(train_scaled, train_target)
    
    print(lasso.score(train_scaled, train_target))
    print(lasso.score(test_scaled, test_target))
  ```

  - 과대적합을 잘 억제하고 테스트 성능을 크게 높임
<br>

- 라쏘 모델은 계수 값을 아예 0으로 만들 수 있으며 coef_속성에 저장되어 있음
- 이 중 0인 것을 헤아려 보기

  ```python
    print(np.sum(lasso.coef_ == 0))  #np.sum() 함수는 배열을 모두 더한 값을 반환
  ```

    - 결과가 40이 나왔기 때문에 55개의 특성을 모델에 주입했지만 라쏘 모델이 사용한 특성은 15개임   
    -> 이러한 특징으로 라쏘 모델을 유용한 특성을 골라내는 용도로 사용 가능
  <br><br><br><br>    


### 최종 코드

3-3.ipynb : [github](https://github.com/rickiepark/hg-mldl/blob/master/3-3.ipynb)
<br><br><br>

### 핵심 패키지와 함수
- pandas
  - read_csv()
- scikit-learn
  - PolynomialFeatures
  - Ridge
  - Lasso
    


