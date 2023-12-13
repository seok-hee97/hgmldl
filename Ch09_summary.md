# Chapter09 텍스트를 위한 인공 신경망
## 한빛 마켓의 댓글을 분석하라!
<br>
<br>

## 09-1 순차 데이터와 순환 신경망

### 순차 데이터
- 텍스트나 **시계열 데이터**와 같이 순서에 의미가 있는 데이터   
  > 시계열 데이터 : 일정한 시간 간격으로 기록된 데이터
  
  <img src="https://github.com/piolink/pyree/assets/67042526/1522ba4b-6e4a-4eb4-90a9-5fe9f788779c"  width="350">
<br>

- 지금까지 사용한 데이터들은 순서와는 상관이 없었음
  - 어떤 샘플이 먼저 주입되어도 모델의 학습에 큰 영향을 미치지 않음
<br>

- 텍스트 데이터는 단어의 순서가 중요한 순차 데이터   
  - 이러한 데이터는 순서를 유지하며 신경망에 주입   
    -> 순차 데이터를 다룰 때는 이전에 입력한 데이터를 기억하는 기능이 필요   
        ex) 별로지만 추천해요 : 추천해요(긍정) + "별로지만"(부정) => 무조건 긍정적 X
  <br>

- 완전 연결 신경망이나 합성곱 신경망은 이런 기억 장치가 없음   
  - 하나의 샘플(하나의 배치)을 사용하여 정방향 계산을 수행하고 나면 그 샘플은 버려지고 재사용하지 않음
  - 입력 데이터의 흐름이 앞으로만 전달되는 신경망을 **피드포워드 신경망**

    <img src="https://github.com/piolink/pyree/assets/67042526/3161208d-eceb-4a44-9a81-8d97583530a2"  width="500">
<br>

- 다음 샘플을 위해서 이전 데이터가 신경망 층에 순환될 필요가 있으며 이를 **순환 신경망**이라고 함
<br><br><br>


### 순환 신경망
- 일반적인 완전 연결 신경망과 거의 비슷하며 완전 연결 신경망에 이전 데이터의 처리 흐름을 순환하는 고리 하나를 추가

  <img src="https://github.com/piolink/pyree/assets/67042526/a26363b5-811c-4348-b920-93ef8c325c50"  width="200">
  
  > 은닉층에 있는 붉은 고리 -> 뉴런의 출력이 다시 자기 자신으로 전달(재사용)
  <br>

  - (예시) A, B, C 3개의 샘플을 처리하는 순환 신경망의 뉴런이 존재, O는 출력된 결과   
    - 첫 번째 샘플 A를 처리하고 난 출력(O<sub>A</sub>)이 다시 뉴런으로 들어감

      <img src="https://github.com/piolink/pyree/assets/67042526/4c2a4b19-fd68-448c-83fb-2eb2b2162a3f"  width="250">

    - 그다음 B를 처리할 때 앞에서 A를 사용해 만든 출력 O<sub>A</sub>를 함께 사용
      
      <img src="https://github.com/piolink/pyree/assets/67042526/1f66e4a1-72cb-4662-9e03-152c57717bfd"  width="250">

    - 따라서 O<sub>A</sub>와 B를 사용해서 만든 O<sub>B</sub>에는 A에 대한 정보가 어느 정도 포함되어 있음, 그다음 C를 처리할 때 O<sub>B</sub>를 함께 사용

      <img src="https://github.com/piolink/pyree/assets/67042526/14b4605e-3e4a-4949-85fd-455c3ca979df"  width="250">

        -> 결과적으로 O<sub>C</sub>에 포함된 정보들은 O<sub>B</sub>를 사용했기 때문에 B와, O<sub>A</sub>도 사용했기 때문에 A에 대한 정보도 담겨 있음,   
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;즉 O<sub>C</sub>에 B와 A에 대한 정보가 담겨 있음
      <br><br>

- 샘플을 처리하는 한 단계를 **타임스텝**이라고 함
- 순환 신경망은 이전 타임스텝의 샘플을 기억하지만 타임스텝이 오래될수록 순환되는 정보는 희미해짐
<br>

- 순환 신경망에서는 특별히 층을 **셀**이라고 함
- 한 셀에는 여러 개의 뉴런이 있지만 완전 연결 신경망과 달리 뉴런을 모두 표시하지 않고 하나의 셀로 층을 표현
- 셀의 출력을 **은닉 상태**라고 부름

  <img src="https://github.com/piolink/pyree/assets/67042526/cd99eba4-f858-4896-adc8-b3c75ba95647"  width="300">

  > 중요한 것은 층의 출력(은닉 상태)을 다음 타임 스텝에 재사용
<br>

- 일반적으로 은닉층의 활성화 함수로 하이퍼볼릭 탄젠트 함수인 tanh가 많이 사용됨
- tanh 함수도 S자 모양을 띠기 때문에 종종 시그모이드 함수라고도 부름   
  - 시그모이드 함수(0-1)와는 달리 -1~1 사이의 범위를 가짐

    <img src="https://github.com/piolink/pyree/assets/67042526/56bfa413-c9f8-4a04-9a26-2a2500c406ec"  width="260">

    > 순환 신경망 그림에서 번거로움을 피하기 위해 활성화 함수를 표시하지 않는 경우가 많지만,   
    > 순환 신경망에서 활성화 함수가 반드시 필요함
<br>

- 피드포워드 신경망과 같이 순환 신경망에서도 뉴런은 입력과 가중치를 곱함
- 다만 순환 신경망의 뉴런은 가중치가 하나 더 있는데 이전 타임스텝의 은닉 상태에 곱해지는 가중치
- 셀은 입력과 이전 타임스텝의 은닉 상태를 사용하여 현재 타임스텝의 은닉 상태를 만듬
<br>

- 아래의 그림에서 2개의 가중치를 셀 안에 구분해서 표시
- W<sub>x</sub>는 입력에 곱해지는 가중치, W<sub>h</sub>는 이전 타임스텝의 은닉 상태에 곱해지는 가중치   
  > 뉴런마다 하나의 절편이 포함(표시는 안함)

  <img src="https://github.com/piolink/pyree/assets/67042526/a5e46c44-cea6-4d46-b6bb-4e1ee9d52667"  width="290">

- 셀의 출력이 다음 타임스탭에 재사용되기 때문에 타임스텝으로 셀을 나누어 그릴 수 있음

  <img src="https://github.com/piolink/pyree/assets/67042526/f16f7004-b115-4294-8ae4-c425571cfb21"  width="550">

  
- 위의 타임스텝으로 펼친 부분에서 알 수 있는 것은 모든 타임스텝에서 사용되는 가중치는 W<sub>h</sub> 하나임
- 가중치는 W<sub>h</sub>는 타임스텝에 따라 변화되는 뉴런의 출력을 학습 -> 순차 데이터를 다루는데 필요
- 맨 처음 타임스텝 1에서 사용되는 이전 은닉 상태 h<sub>0</sub>의 값은 0임   
  -> 맨 처음 샘플을 입력할 때는 이전 타임스텝이 없기 때문에 h<sub>0</sub>은 모두 0으로 초기화
<br><br><br>


### 셀의 가중치와 입출력

- 복잡한 모델을 배울수록 가중치 개수를 계산해보면 잘 이해하고 있는지 알 수 있음
<br>

- (예시) 순환층에 입력되는 특성의 개수가 4개, 순환층의 뉴런이 3개

  - 순환층의 가중치 크기
    - W<sub>x</sub>의 크기   
      > 입력층과 순환층의 뉴런이 모두 완전 연결되기 때문에 4x3=12개가 됨
  
      <img src="https://github.com/piolink/pyree/assets/67042526/ef272c3c-1590-46ba-b852-bf2483e992b5"  width="230">
  
      
    - 순환층에서 다음 타임스텝에 재사용되는 은닉 상태를 위한 가중치 W<sub>b</sub>의 크기   
      > 순환층의 첫 번째 뉴런(r<sub>1</sub>)의 은닉 상태가 다음 타임스텝에 재사용될 때 모두 전달됨,   
      > 즉, 이전 타임스텝의 은닉 상태는 다음 타임스텝의 뉴런에 완전히 연결   
      > 따라서 이 순환층의 은닉 상태를 위한 가중치 W<sub>b</sub>는 3x3=9개
  
      <img src="https://github.com/piolink/pyree/assets/67042526/b569a533-2df9-4e4b-a9f2-5fcc7e12ee0b"  width="150">
    
  
    - 모델 파라미터 개수 계산   
      > 가중치 + 절편   
      > 각 뉴런마다 하나의 절편이 존재, 따라서 이 순환층은 12+9+3 = 24개의 모델 파라미터를 가짐
  
        - 모델 파라미터 수 = W<sub>x</sub>+W<sub>b</sub>+절편 = 12+9+3 = 24
      <br><br>

  - 순환층의 입출력
    - 순환층은 일반적으로 샘플마다 2개의 차원을 가짐   
      > 보통 하나의 샘플을 하나의 시퀀스라고 함   
      > 시퀀스 안에는 여러 개의 아이템이 들어 있으며 시퀀스의 길이가 타임스텝 길이가 됨

        - (예시) 샘플은 4개의 단어로 이루어져 있으며, 각 단어를 3개의 어떤 숫자로 표현

          <img src="https://github.com/piolink/pyree/assets/67042526/a5409543-7437-454d-a0af-40080212f128"  width="450">

            - 이런 입력이 순환층을 통과하면 두 번째, 세 번째 차원이 사라지고 순환층의 뉴런 개수만큼 출력됨

              <img src="https://github.com/piolink/pyree/assets/67042526/78ea36d4-7f33-4274-b2ac-8a298d298a54"  width="370">


    - 앞에서는 셀이 모든 타임스텝에서 출력을 만든 것처럼 표현했지만, 사실 순환층은 기본적으로 마지막 타임스텝의 은닉 상태만 출력으로 내보냄
    - 순환 신경망도 여러 개의 층을 쌓을 수 있으며 순환층을 여러개 쌓았을 때의 셀의 출력은 마지막 셀을 제외한 다른 모든 셀은 모든 타임스텝의 은닉 상태를 출력   
      > 셀의 입력은 샘플마다 타임스텝과 단어 표현으로 이루어진 2차원 배열이어야 하기 때문에 첫 번째 셀이 마지막 타임스텝의 은닉 상태만 출력하면 안됨

      <img src="https://github.com/piolink/pyree/assets/67042526/8e24bc37-f00a-4ac7-b7af-dbe047ae2b68"  width="350">

      - 2개의 순환층을 쌓은 경우
   <br><br>   

  - 출력층의 구성
    - 순환 신경망도 마지막에는 밀집층을 두어 클래스를 분류
    - 다중 분류일 경우에는 출력층에 클래스 개수만큼 뉴런을 두고 소프트맥스 활성화 함수를 사용
    - 이진 분류일 경우에는 하나의 뉴런을 두고 시그모이드 활성화 함수를 사용

    - 합성곱 신경망과 다른 점은 마지막 셀의 출력이 1차원이기 때문에 Flatten 클래스로 펼칠 필요가 없어 셀의 출력을 그대로 밀집층에 사용 가능
      - (예시) 다중 분류 문제에서 입력 샘플의 크기가 (20, 100)일 경우 하나의 순환층을 사용하는 순환 신경망의 구조

        <img src="https://github.com/piolink/pyree/assets/67042526/0ea948ae-3309-45fa-9fff-8b07ed72691f"  width="500">
<br><br><br><br><br><br><br>



## 09-2 순환 신경망으로 IMDB 리뷰 분류하기

- 대표적인 순환 신경망 문제인 IMDB 리뷰 데이터셋을 사용해 가장 간단한 순환 신경망 모델을 훈련
- 데이터셋을 두 가지 방법으로 변형하여 순환 신경망에 주입 -> 원-핫 인코딩, 단어 임베딩
<br><br>


### IMDB 리뷰 데이터셋

- 유명한 인터넷 영화 데이터베이스인 imdb.com에서 수집한 리뷰를 감상평에 따라 긍정과 부정으로 분류해 놓은 데이터셋
- 총 50,000개의 샘플로 이루어져 있고 훈련 데이터와 테스트 데이터에 각각 25,000개씩 나누어져 있음
<br>

- **자연어 처리**는 컴퓨터를 사용해 인간의 언어를 처리하는 분야(음성 인식, 기계 번역, 감성 분석 등)
- 자연어 처리 분야에서는 훈련 데이터를 종종 **말뭉치**라고 부름
<br>

- 텍스트 자체를 신경망에 전달하지 않고 텍스트 데이터의 경우 단어를 숫자 데이터로 바꾸는 일반적인 방법은 데이터에 등장하는 단어마다 고유한 정수를 부여

  <img src="https://github.com/piolink/pyree/assets/67042526/f7fa7fbd-e2cd-47db-8ff6-3f6bd63cadce"  width="500">

  - 각 단어를 하나의 정수에 매핑하며 동일한 단어는 동일한 정수에 매핑
  - 단어에 매핑되는 정수는 단어의 의미나 크기와 관련이 없음
  - 일반적으로 영어 문장은 모두 소문자로 바꾸고 구둣점을 삭제한 다음 공백을 기준으로 분리
  - 분리된 단어를 **토큰**이라고 하며 1개의 토큰이 하나의 타임스텝에 해당
  - 일반적으로 한글은 형태소 분석을 통해 토큰을 만들며 KoNLPy를 사용함
<br>

- 토큰에 할당하는 정수 중에 특정한 용도로 예약되어 있는 경우   
  -> 0:패딩, 1:문장의 시작, 2:어휘 사전에 없는 토큰   
    > 어휘 사전 : 훈련 세트에서 고유한 단어를 뽑아 만든 목록
  <br>

- 텐서플로에 이미 정수로 바꾼 데이터가 포함되어 있기 때문에 tensorflow.keras.datasets 패키지 아래 imdb 모듈 사용
- 전체 데이터셋에서 가장 자주 등장하는 단어 300개만 사용

  ```python
    from tensorflow.keras.datasets import imdb
    
    (train_input, train_target), (test_input, test_target) = imdb.load_data(
        num_words=300)

    #훈련 세트, 테스트 세트의 크기
    print(train_input.shape, test_input.shape)
  ```

  > (25000,) (25000,)

  - IMDB 리뷰 텍스트는 길이가 제각각이기 때문에 고정 크기의 2차원 배열에 담기 보다는 리뷰마다 별도의 파이썬 리스트로 담아야 메모리를 효율적으로 사용 가능

    <img src="https://github.com/piolink/pyree/assets/67042526/56a4e40f-1aa4-4e1a-9501-a57efb02aefc"  width="300">

    - 이 데이터는 개별 리뷰를 담은 파이썬 리스트 객체로 이루어진 넘파이 배열
<br>

- 첫 번째 리뷰와 두 번째 리뷰의 길이

  ```python
    print(len(train_input[0]))
    print(len(train_input[1]))
  ```

  > 218   
  > 189
<br>

- 첫 번째 리뷰에 담긴 내용 출력
  
  ```python
    print(train_input[0])
  ```

  > [1, 14, 22, 16, 43, 2, 2, 2, 2, 65, 2, 2, 66, 2, 4, 173, 36, 256, 5, 25, 100, 43, 2, 112, 50, 2, 2, 9, 35, 2, 284, 5, 150, 4, 172, 112, 167, 2, 2, 2, 39, 4, 172, 2, 2, 17, 2, 38, 13, 2, 4, 192, 50, 16, 6, 147, 2, 19, 14, 22, 4, 2, 2, 2, 4, 22, 71, 87, 12, 16, 43, 2, 38, 76, 15, 13, 2, 4, 22, 17, 2, 17, 12, 16, 2, 18, 2, 5, 62, 2, 12, 8, 2, 8, 106, 5, 4, 2, 2, 16, 2, 66, 2, 33, 4, 130, 12, 16, 38, 2, 5, 25, 124, 51, 36, 135, 48, 25, 2, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 2, 16, 82, 2, 8, 4, 107, 117, 2, 15, 256, 4, 2, 7, 2, 5, 2, 36, 71, 43, 2, 2, 26, 2, 2, 46, 7, 4, 2, 2, 13, 104, 88, 4, 2, 15, 297, 98, 32, 2, 56, 26, 141, 6, 194, 2, 18, 4, 226, 22, 21, 134, 2, 26, 2, 5, 144, 30, 2, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 2, 88, 12, 16, 283, 5, 16, 2, 113, 103, 32, 15, 16, 2, 19, 178, 32]

  - 어휘 사전에는 300개의 단어만 들어가 있어 어휘 사전에 없는 단어는 모두 2로 표시됨
 <br>

- 타깃 데이터 출력

   ```python
     print(train_target[:20])
   ```

    > [1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]

    - 타깃값 - 0:부정, 1:긍정
  <br>

- 훈련 세트에서 검증 세트(20%) 분리

  ```python
    from sklearn.model_selection import train_test_split
    
    train_input, val_input, train_target, val_target = train_test_split(
        train_input, train_target, test_size=0.2, random_state=42)
  ```
<br>

- 평균적인 리뷰의 길이와 가장 짧은 리뷰의 길이, 가장 긴 리뷰의 길이 확인   
  -> 넘파이 리스트 내포를 사용해 train_input의 원소를 순회하면서 길이 측정

  ```python
    import numpy as np
    
    lengths = np.array([len(x) for x in train_input])

    #리뷰 길이의 평균, 중간값 구하기
    print(np.mean(lengths), np.median(lengths))
  ```

  > 239.00925 178.0

  - 히스토그램으로 표현

    ```python
      import matplotlib.pyplot as plt
      
      plt.hist(lengths)
      plt.xlabel('length')
      plt.ylabel('frequency')
      plt.show()
    ```
    
    <img src="https://github.com/piolink/pyree/assets/67042526/91909b70-00f2-43ff-a959-e0cdd47c7cc5"  width="400">
  <br>

  
- pad_sequences() 함수는 기본으로 maxlen보다 긴 시퀀스의 앞부분을 자름
- 일반적으로 시퀀스의 뒷부분의 정보가 더 유용하리라 기대하기 때문
- 만약 시퀀스의 뒷부분을 잘라내고 싶다면 pad_sequences() 함수의 truncating 매개변수의 값을 기본값 'pre'가 아닌 'post'로 바꾸면 됨
<br>

- 패딩 토큰은 시퀀스의 뒷부분이 아니라 앞부분에 추가됨, 시퀀스의 마지막에 있는 단어가 셀의 은닉 상태에 가장 큰 영향을 미치게 되므로 마지막에 패딩을 추가하는 것을 일반적으로 선호하지 않음
- 하지만 샘플의 뒷부분에 패딩을 추가하길 원한다면 pad_sequences() 함수의 padding 매개변수의 기본값 'pre'가 아닌 'post'로 바꾸면 됨
<br>

- 훈련 세트와 검증 세트의 길이를 100으로 맞추기

  ```python
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    train_seq = pad_sequences(train_input, maxlen=100)
    val_seq = pad_sequences(val_input, maxlen=100)
  ```  
<br><br>


### 순환 신경망 만들기

- 케라스는 여러 종류의 순환층 클래스를 제공하며 그 중 가장 간단한 것은 SimpleRNN 클래스 
- 케라스의 Sequential 클래스로 만든 신경망 코드
  
  ```python
    from tensorflow import keras
    
    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(8, input_shape=(100, 300)))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
  ```

  > 전과 달라진 것은 Dense, Conv2D 클래스 대신 SimpleRNN 클래스를 사용   
  > 첫 번째 매개변수에는 사용할 뉴런의 개수를 지정, input_shape에 입력 차원을 (100, 300)으로 지정
  > 순환층도 당연히 활성화 함수를 사용해야하며 SimpleRNN 클래스의 activation 매개변수의 기본값은 'tanh'로 하이퍼볼릭 탄젠트 함수를 사용
  >
  > 이전 섹션에서 만든 train_seq, val_seq에는 한 가지 큰 문제가 있는데 토큰을 정수를 변환한 이 데이터를 신경망에 주입하면 큰 정수가 큰 활성화 출력을 만듬
  > 하지만 이 정수 사이에는 관련이 없음
<br>

- 정숫값에 있는 크기 속성을 없애고 각 정수를 고유하게 표현하는 방법은 원-핫 인코딩   
  > 원-핫 인코딩은 정숫값을 배열에서 해당 정수 위치의 원소만 1이고 나머지는 모두 0으로 변환함
  - (예시) train_seq[0]의 첫 번째 토큰인 10을 원-핫 인코딩으로 변환

    <img src="https://github.com/piolink/pyree/assets/67042526/4b3ad649-f26e-47b2-8aaf-0f2f307ee577"  width="550">
  <br>

- imdb.load_data() 함수에서 300개의 단어만 사용하도록 지정했기 때문에 고유한 단어는 모두 300개
- 즉 훈련 데이터에 포함될 수 있는 정숫값의 범위는 0(패딩 토큰)에서 299까지
- 따라서 이 범위를 원-핫 인코딩으로 표현하려면 배열의 길이가 300이어야 함
<br>

- 토큰마다 300개의 숫자를 사용해 표현하고, 다만 300개 중에 하나만 1이고 나머지는 모두 0으로 만들어 정수 사이에 있던 크기 속성을 없애는 원-핫 인코딩을 사용
- 케라스에서 원-핫 인코딩을 위한 유틸리티를 제공하며 keras.utils 패키지 아래에 있는 to_categorical() 함수

  ```python
    #train_seq를 원-핫 인코딩으로 변환하여 train_oh 배열 생성
    train_oh = keras.utils.to_categorical(train_seq)

    #배열의 크기 출력
    print(train_oh.shape)
  ```

  > (20000, 100, 300)
<br>

- train_oh의 첫 번째 샘플의 첫 번째 토큰 10이 잘 인코딩되었는지 확인

  ```python
    print(train_oh[0][0][:12])

    #모든 원소의 값을 더해서 1이 되는지 확인
    print(np.sum(train_oh[0][0]))
  ```

  > [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]   
  > 1.0
<br>

- val_seq도 원-핫 인코딩으로 변환

  ```python
    val_oh = keras.utils.to_categorical(val_seq)
  ```
<br>

- 앞서 만든 모델의 구조 출력

  ```python
    model.summary()
  ```

  ```
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    simple_rnn (SimpleRNN)      (None, 8)                 2472      
                                                                     
    dense (Dense)               (None, 1)                 9         
                                                                     
    =================================================================
    Total params: 2,481
    Trainable params: 2,481
    Non-trainable params: 0
    _________________________________________________________________
  ```

  - SimpleRNN에 전달할 샘플의 크기는 (100, 300)이지만 이 순환층은 마지막 타임스텝의 은닉 상태만 출력하기 때문에 출력 크기가 순환층의 뉴런 개수와 동일한 8임을 확인
  - 순환층에 사용된 모델 파라미터 개수는 총 300x8 = 2,400개의 가중치와 8x8 = 64개의 가중치를 합친 2,400+64+8 = 2,472개   
    > - 입력 토큰은 300차원의 원-핫 인코딩 배열이 순환층의 뉴런 8개와 완전히 연결   
    > - 순환층의 은닉 상태는 다시 다음 타임스텝에 사용되기 위해 또 다른 가중치와 곱해지며 이 은닉 상태도 순환층의 뉴런과 완전히 연결 => 은닉 상태 크기 x 뉴런 개수    
    > - 마지막으로 뉴런마다 하나의 절편이 있음

<br><br>



### 순환 신경망 훈련하기

- 다른 신경망들과 모델을 만드는 것은 달라도 훈련하는 방법은 모두 같음

- 기본 RMSprop의 학습률 0.001을 사용하지 않기 위해 별도의 RMSprop 객체를 만들어 학습률을 0.0001로 지정
- 에포크 횟수를 100으로 늘리고 배치 크기는 64개로 설정

  ```python
    rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
    model.compile(optimizer=rmsprop, loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5',
                                                    save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                      restore_best_weights=True)
    
    history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
                        validation_data=(val_oh, val_target),
                        callbacks=[checkpoint_cb, early_stopping_cb])
  ```

  ```
    Epoch 1/100
    313/313 [==============================] - 31s 81ms/step - loss: 0.7003 - accuracy: 0.5002 - val_loss: 0.6970 - val_accuracy: 0.5058
    Epoch 2/100
    313/313 [==============================] - 24s 78ms/step - loss: 0.6956 - accuracy: 0.5123 - val_loss: 0.6946 - val_accuracy: 0.5124
    Epoch 3/100
    313/313 [==============================] - 24s 76ms/step - loss: 0.6917 - accuracy: 0.5282 - val_loss: 0.6909 - val_accuracy: 0.5318
    Epoch 4/100
    313/313 [==============================] - 24s 77ms/step - loss: 0.6844 - accuracy: 0.5549 - val_loss: 0.6833 - val_accuracy: 0.5690
    Epoch 5/100
    313/313 [==============================] - 24s 76ms/step - loss: 0.6784 - accuracy: 0.5778 - val_loss: 0.6797 - val_accuracy: 0.5770
    ...
    Epoch 35/100
    313/313 [==============================] - 24s 75ms/step - loss: 0.5040 - accuracy: 0.7640 - val_loss: 0.5217 - val_accuracy: 0.7472
    Epoch 36/100
    313/313 [==============================] - 24s 76ms/step - loss: 0.5023 - accuracy: 0.7645 - val_loss: 0.5195 - val_accuracy: 0.7456
    Epoch 37/100
    313/313 [==============================] - 24s 76ms/step - loss: 0.5008 - accuracy: 0.7658 - val_loss: 0.5200 - val_accuracy: 0.7474
    Epoch 38/100
    313/313 [==============================] - 24s 78ms/step - loss: 0.4994 - accuracy: 0.7647 - val_loss: 0.5198 - val_accuracy: 0.7484
    Epoch 39/100
    313/313 [==============================] - 25s 80ms/step - loss: 0.4982 - accuracy: 0.7648 - val_loss: 0.5206 - val_accuracy: 0.7472
  ```

  - 35번째 에포크에서 조기 종료되었으며 검증 세트에 대한 정확도는 약 80% 정도
  - 훈련 손실과 검증 손실을 그래프로 생성

    ```python
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.xlabel('epoch')
      plt.ylabel('loss')
      plt.legend(['train', 'val'])
      plt.show()
    ```

    <img src="https://github.com/piolink/pyree/assets/67042526/a9365968-770d-4493-b939-5c564ed99509"  width="400">

    - 훈련 손실은 꾸준히 감소하고 있지만 검증 손실은 대략 20번째 에포크에서 감소가 둔해짐
    
    <br>

- 원-핫 인코딩의 단점은 입력 데이터가 엄청 커짐 -> 훈련 데이터가 커질수록 더 문제가 됨

<br><br>



### 단어 임베딩을 사용하기

- **단어 임베딩**은 순환 신경망에서 텍스트를 처리할 때 즐겨 사용하는 방법으로, 각 단어를 고정된 크기의 실수 벡터로 바꾸어 줌
    
  <img src="https://github.com/piolink/pyree/assets/67042526/29bc7f43-ae57-465b-bdcb-135054202746"  width="400">
<br>

- 이런 단어 임베딩으로 만들어진 벡터는 원-핫 인코딩된 벡터보다 훨씬 의미 있는 값으로 채워져 있기 때문에 자연어 처리에서 더 좋은 성능을 내는 경우가 많음
- 케라스에서 keras.layers 패키지 아래 Embedding 클래스로 임베딩 기능을 제공
- 단어 임베딩의 장점은 입력으로 정수 데이터를 받음 -> 메모리를 훨씬 효율적으로 사용

  ```python
    model2 = keras.Sequential()
    
    model2.add(keras.layers.Embedding(300, 16, input_length=100))
    model2.add(keras.layers.SimpleRNN(8))
    model2.add(keras.layers.Dense(1, activation='sigmoid'))

    #모델 구조 출력
    model2.summary()
  ```

  ```
    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, 100, 16)           4800      
                                                                     
     simple_rnn_1 (SimpleRNN)    (None, 8)                 200       
                                                                     
     dense_1 (Dense)             (None, 1)                 9         
                                                                     
    =================================================================
    Total params: 5,009
    Trainable params: 5,009
    Non-trainable params: 0
    _________________________________________________________________
  ```

  - 모델 파라미터 개수 = (16x8 = 128) + (8x8 = 64) + 8 = 200개
<br>

- 모델 훈련

  ```python
    rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
    model2.compile(optimizer=rmsprop, loss='binary_crossentropy',
                   metrics=['accuracy'])
    
    checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.h5',
                                                    save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                      restore_best_weights=True)
    
    history = model2.fit(train_seq, train_target, epochs=100, batch_size=64,
                         validation_data=(val_seq, val_target),
                         callbacks=[checkpoint_cb, early_stopping_cb])

  ```

  ```
    Epoch 1/100
    313/313 [==============================] - 42s 127ms/step - loss: 0.6893 - accuracy: 0.5351 - val_loss: 0.6706 - val_accuracy: 0.5872
    Epoch 2/100
    313/313 [==============================] - 36s 116ms/step - loss: 0.6399 - accuracy: 0.6467 - val_loss: 0.6234 - val_accuracy: 0.6664
    Epoch 3/100
    313/313 [==============================] - 36s 114ms/step - loss: 0.6051 - accuracy: 0.6941 - val_loss: 0.6003 - val_accuracy: 0.6948
    Epoch 4/100
    313/313 [==============================] - 37s 118ms/step - loss: 0.5831 - accuracy: 0.7172 - val_loss: 0.5888 - val_accuracy: 0.7026
    Epoch 5/100
    313/313 [==============================] - 37s 117ms/step - loss: 0.5663 - accuracy: 0.7305 - val_loss: 0.5669 - val_accuracy: 0.7300
    ...
    Epoch 20/100
    313/313 [==============================] - 37s 117ms/step - loss: 0.4880 - accuracy: 0.7740 - val_loss: 0.5124 - val_accuracy: 0.7500
    Epoch 21/100
    313/313 [==============================] - 36s 115ms/step - loss: 0.4871 - accuracy: 0.7749 - val_loss: 0.5143 - val_accuracy: 0.7488
    Epoch 22/100
    313/313 [==============================] - 36s 114ms/step - loss: 0.4847 - accuracy: 0.7775 - val_loss: 0.5148 - val_accuracy: 0.7486
    Epoch 23/100
    313/313 [==============================] - 36s 116ms/step - loss: 0.4830 - accuracy: 0.7775 - val_loss: 0.5135 - val_accuracy: 0.7500
  ```

<br>

- 훈련 손실과 검증 손실
  
  ```python
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'val'])
    plt.show()
  ```

  <img src="https://github.com/piolink/pyree/assets/67042526/9a76783c-d6bf-4712-b815-70bf705bf4d0"  width="400">

  - 검증 손실이 더 감소되지 않아 훈련이 적절히 조기 종료된것 같지만 훈련 손실은 계속 감소하여 개선해야 됨

<br><br><br><br>





### 최종 코드

9-2.ipynb : [github](https://github.com/rickiepark/hg-mldl/blob/master/9-2.ipynb)
<br><br><br>


### 핵심 패키지와 함수
- TensorFlow
  - pad_sequences()
  - to_categorical()
  - SimpleRNN
  - Embedding
    
<br><br><br><br><br><br><br>


