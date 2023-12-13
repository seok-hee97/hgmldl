# [Chapter 6] 비지도 학습

## Ch06-1 군집 알고리즘

> 흑백 사진을 분류하기 위해 여러 가지 아이디어를 내면서 비지도 학습과 군집 알고리즘에 대해 이해합니다.

### 타깃을 모르는 비지도 학습

타깃을 모르는 사진을 종류별로 분류하려고 하는데 타깃을 모를때는 기존의 알고리즘과 다른 알고리즘을 사용해야 합니다. 이렇게 타깃이 없을 때 사용하는 머신러닝 알고리즘이 바로 <span style="color:violet"> 비지도 학습 </span> 입니다.

> 사람이 가르쳐 주지 않아도 데이터에 있는 무언가를 학습하는 것이라고 <br/> 생각하면 됩니다.

<hr/>
  
  ### 예시 - 과일 데이터 준비하기

  > 먼저 <span style="color:blue"> numpy </span> 와 <span style="color:blue"> matplotlib </span> 패키지들을 import 합니다.

```python

import numpy as np
import matplotilb.pyplot as plt

```
> numpy에서 __npy__ 로드하는 방법은 아주 간단합니다. __load()__ 메서드에 파일 이름을 전달하는것이 전부다. 

```python

fruits = np.load('fruits_300.npy')

```
> **fruits**는 넘파이 배열이고 fruits_300.npy 파일에 들어 있는 모든 데이터를 담고 있습니다. 

```python
print(fruits.shape)
```
> 결과물: (300, 100, 100)


**matploblib**의 imshow() 함수를 사용하면 넘파이 배열로 저장된 이미지를 쉽게 그릴 수 있습니다. 흑백 이미지이므로 cmap 매개변수를 'gray'로 지정합니다.

```python

plt.imshow(fruits[0], cmap='gray')
plt.show()
```
<pre>
<img width="307" alt="image" src="https://github.com/piolink/pyree/assets/75467180/21a84c92-cffc-4345-8b63-c355f970e785"/>

</pre>

첫번째 이미지는 사과인거 같습니다. 
<br/>
우리의 관심 대상은 바탕이 아니라 사과입니다. 흰색 바탕은 중요하지 않지만 컴퓨터에게는 중요한 부분입니다. 바탕을 검게 만들고 사진에 짙게 사과를 강조해야 합니다.

> 컴퓨터는 왜 255에 가까운 바탕에 집중하는 것일까..?
>   > 알고리즘이 어떤 출력을 만들기 위해 곱셈, 덧셈을 한다. 픽셀값이 0이면 출력도 0이 된다. 픽셀값이 높으면 출력값도 커진다.


따라서 이미지를 아래와 같이 만든다.  <br/>

위 그림에서 밝은 부분이 0에 가깝고 짙은 부분이 255에 가까운 값이다. 

```python
fig, axs = plt.subplots(1, 2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()
```

* **matploblib**의 **subplots()** 함수를 사용하면 여러 개의 그래프를 배열처럼 쌓을 수 있도록 함. 
  

### 픽셀값 분석하기

fruits 배열에서 순서대로 100개씩 선택하기 위해 슬라이싱 연산자를 사용합니다.

```python
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)
```
```python
print(apple.shape)
```
> 결과: (100, 10000)

apple, pineapple, banna 배열에 들어 있는 샘플의 픽셀 평균값을 계산해볼때 **mean()** 메서드를 사용할 것입니다. 픽셀의 평균값을 계산헤야 합니다. **mean()** 메서드가 평균을 계산할 축을 지정할 때 **axis=0**으로 하면 첫 번째 축인 행을 따라 계산합니다. **axis=1**은 두 번째 축이 될 것입니다.

```python
print(apple.mean(axis=1))
```

### **결과**
<pre> 
[ 88.3346  97.9249  87.3709  98.3703  92.8705  82.6439  94.4244  95.5999
  90.681   81.6226  87.0578  95.0745  93.8416  87.017   97.5078  87.2019
  88.9827 100.9158  92.7823 100.9184 104.9854  88.674   99.5643  97.2495
  94.1179  92.1935  95.1671  93.3322 102.8967  94.6695  90.5285  89.0744
  97.7641  97.2938 100.7564  90.5236 100.2542  85.8452  96.4615  97.1492
  90.711  102.3193  87.1629  89.8751  86.7327  86.3991  95.2865  89.1709
  96.8163  91.6604  96.1065  99.6829  94.9718  87.4812  89.2596  89.5268
  93.799   97.3983  87.151   97.825  103.22    94.4239  83.6657  83.5159
 102.8453  87.0379  91.2742 100.4848  93.8388  90.8568  97.4616  97.5022
  82.446   87.1789  96.9206  90.3135  90.565   97.6538  98.0919  93.6252
  87.3867  84.7073  89.1135  86.7646  88.7301  86.643   96.7323  97.2604
  81.9424  87.1687  97.2066  83.4712  95.9781  91.8096  98.4086 100.7823
 101.556  100.7027  91.6098  88.8976]
</pre>

히스토그램을 그려보면 평균값이 어떻게 분포되어 있는지 한눈에 잘 볼 수 있습니다.
<br/> **히스토그램이란?** 값이 발생한 빈도를 그래프로 표시한 것


**matploblib**의 **hist()** 함수를 사용해서 히스토그램을 그려보도록 합시다.

```python
plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()
```

<pre>
결과
<img width="424" alt="image" src="https://github.com/piolink/pyree/assets/75467180/27ffed45-16d7-4455-a95c-14902eb9301a">

</pre>

이제는 막대그래프를 그려보도록 합니다.

```python
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()
```
<pre>
결과
<img width="924" alt="image" src="https://github.com/piolink/pyree/assets/75467180/0fafde59-541f-45c9-aecf-862add5b688d">
</pre>

> 3개의 그래프를 보면 과일마다 값이 높은 구간이 다릅니다. 


```python
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()
```

<pre>
결과
<img width="935" alt="image" src="https://github.com/piolink/pyree/assets/75467180/798a1035-aaf0-4b6c-b0bf-87f08ff670af">
</pre>

> 세 과일은 픽셀 위치에 따라 값의 크기가 차이 납니다. 

## 평균값과 가까운 사진 고르기

```python
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1,2))
print(abs_mean.shape)
```
<pre>
결과
(300, )
</pre>

가장 작은 순서대로 1000개를 골라 보겠습니다. 
```python
apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
        axs[i, j].axis('off')
plt.show()
```
<pre>
결과
<img width="606" alt="image" src="https://github.com/piolink/pyree/assets/75467180/908fe29c-c30f-40cb-abd5-109ad917e001">
</pre>

> **subplots()** 함수로 10 x 10, 총 100개의 서브 그래프를 만들고 **for 반복문** 을 순회하면서 10개 행과 열에 이미지를 출력합니다. 

이렇게 비슷한 샘플끼리 그룹으로 모으는 작업을 <span style="color:violet"> 군집</span> 이라고 합니다. 군집은 대표적인 비지도 학습 중 하나입니다. 군집 알고리즘에서 만든 그룹을 <span style="color:violet"> 클러스터</span> 라고 부릅니다.


## 핵심
> * **비지도 학습은** 머신러닝의 한 종류로 훈련 데이터에 타깃이 없습니다. 타깃이 없기 때문에 외부의 도움 없이 스스로 유용한 무언가를 학습해야 합니다.
>  * **히스토그램**은 구간별로 값이 발생한 빈도를 그래프로 표시한 것입니다.
>  * **군집**은 비슷한 샘플끼리 하나의 그룹으로 모으는 대표적인 비지도 학습 작업입니다.

## Ch06-2 K-평균

진짜 비지도 학습엣는 사진에 어떤 과일이 들어있는지 알지 못하는 경우가 많습니다.
이런 경우에는 <span style="color:violet">K-평균</span> 군집 알고리즘이 평균값을 자동으로 찾아줍니다.
<br/>이 평균값이 클러스터의 중심에 위치하기 때문에 <span style="color:violet">클러스터 중심 </span> 또는 <span style="color:violet"> 센트로이드 </span> 라고 부릅니다.

### K-평균 알고리즘 소개

k-평균 알고리즘의 작동 방식은 다음과 같습니다.
1. 무작위로 k개의 클러스터 중심을 정합니다.
2. 각 샘플에서 가장 가까운 중심을 정합니다.
3. 클러스터에 속한 샘플의 평균값으로 클러스터 중심을 변경합니다.
4. 클러스터 중심에 변화가 없을 때까지 2번으로 돌아가 반복합니다.


### KMean 클래스 (사이킷런 사용)

```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy
```
> Wget 명령으로 데이터를 다운로드 합니다.

```python
import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)
```
> np.load() 함수를 사용해 넘파이 배열을 준비한 후 3차원 배열을 2차원 배열로 변경합니다.

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
```
> fit() 메서드에서 타깃 데이터를 사용하지 않습니다. <br/>
> 클래스에서 설정할 매개변수는 클러스터 개수를 지정하는 n_clusters입니다

```python
print(km.labels_)
```
<pre> 
결과
[2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 0 2 0 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 0 0 2 2 2 2 2 2 2 2 0 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1]
</pre>

```python
print(np.unique(km.labels_, return_counts=True))
```
<pre>
결과
(array([0, 1, 2], dtype=int32), array([111,  98,  91]))
</pre>


각 클러스터가 어떤 이미지를 나타냈는지 그림으로 출력하기 위해 간단한 유틸리티 함수 **draw-fruits()** 를 만들어 보겠습니다. 

```py
import matplotlib.pyplot as plt

def draw_fruits(arr, ratio=1):
    n = len(arr)    # n은 샘플 개수입니다
    # 한 줄에 10개씩 이미지를 그립니다. 샘플 개수를 10으로 나누어 전체 행 개수를 계산합니다. 
    rows = int(np.ceil(n/10))
    # 행이 1개 이면 열 개수는 샘플 개수입니다. 그렇지 않으면 10개입니다.
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, 
                            figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:    # n 개까지만 그립니다.
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()
```

> km.labels_==0과 같이 쓰면 km.labels_ 배열에서 값이 0인 위치는 True, 그 외는 모두 False가 됩니다. 넘파이는 이런 불리언 배열을 사용해 원소를 선택할 수 있습니다. 이를 <span style="color:violet">불리언 인덱싱 </span> 이라고 합니다. 이를 적용하면 True인 위치의 원소만 모두 추출합니다. 


```py
draw_fruit(fruits[km.labels_==0])
```

<pre>
결과
<img width="681" alt="image" src="https://github.com/piolink/pyree/assets/75467180/b538a378-b3bc-4e91-b580-482fc84acdfa">
</pre>

```py
draw_fruits(fruits[km.labels_==1])
```

<pre>
결과
<img width="644" alt="image" src="https://github.com/piolink/pyree/assets/75467180/58250b57-6003-4885-8dfd-e397bec90f6d">
</pre>

```py
draw_fruits(fruits[km.labels_==2])
```
<pre>
결과
<img width="610" alt="image" src="https://github.com/piolink/pyree/assets/75467180/998ca997-e711-43c4-a5f1-931b12ebcbc1">
</pre>


### 클러스터 중심
KMeans 클래스가 최종적으로 찾은 클러스터 중심은 cluster_centers__ 속성에 저장되어 있습니다. 이 배열은 fruits_2d 샘플의 클러스터 중심이기 때문에 각 중심을 이미지로 출력하려면 100 x 100 크기의 2차원 배열로 바꿔야 합니다.

```py
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)
```
<pre>
결과
<img width="563" alt="image" src="https://github.com/piolink/pyree/assets/75467180/4c4611a4-2e28-46c6-994e-ec86c4958963">
</pre>

**KMeans** 클래스는 훈련 데이터 샘플에서 클러스터 중심까지 중심까지 거리로 변환해주는 **transform()** 메서드가 있다는 것은 마치 **StandardScaler** 클래스처엄 특성값을 변환하는 도구로 사용할 수 있다는 의미입니다.

```py
print(km.transform(fruits_2d[100:101]))
```
<pre> 
결과
[[3393.8136117  8837.37750892 5267.70439881]]
</pre>

```py
print(km.predict(fruits_2d[100:101]))
```
<pre> 
결과
[0]
</pre>

```py
draw_fruits(fruits[100:101])
```
<pre> 
결과
<img width="101" alt="image" src="https://github.com/piolink/pyree/assets/75467180/bface6ed-16b1-423a-b316-054ea0a3f80c">
</pre>


### 최적의 k 찾기
군집 알고리즘에서 적절한 k값을 찾기 위한 완벽한 방법은 없습니다. 적절한 클러스터 개수를 찾기 위한 대표적인 방법인 <span style="color:violet">엘보우 </span> 방법에 대해서 알아보도록 하겠습니다.



```py
inertia = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)

plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()
```
<pre>
결과
<img width="482" alt="image" src="https://github.com/piolink/pyree/assets/75467180/65d8f04a-2c1c-46aa-a3d9-74250a374608">

 </pre>

 > fit() 메서드로 모델을 훈련한 후 inertia_ 속성에 저장된 이너셔 닶을 inertia 리스트에 추가합니다.


 ## Ch06-3 주성분 분석

### 차원과 차원 축소
데이터가 가진 속성을 특성이라고 합니다. 머신러닝에서는 이런 특성을 <span style="color:violet"> 차원 </span> 이라고 부릅니다.
비지도 학습 작업 중 하나인 <span style="color:violet"> 차원 축소 </span> 알고리즘을 다뤄봅시다. 
주성분 분석을 간단히 <span style="color:violet">PCA </span> 라고 부릅니다. 

### 주성분 분석 소개

* 주성분 분석은 데이터에 있는 분산이 큰 방향을 찾는 것으로 이해할 수 있습니다.<br/> 분산이 큰 방향이란 데이터를 잘 표현하는 어떤 백터라고 생각할 수 있습니다.
* 주성분은 원본 차원과 같고 주성분으로 바꾼 데이터는 차원이 줄어든다는 점을 기억해야 합니다. 


### PCA 클래스

```py
import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)
```
> PCA 클래스의 객체를 만들 때 n_components 매개변수에 주성분의 개수를 지정해야 합니다.

```py
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca.fit(fruits_2d)
```
```py
print(pca.components_.shape)
```
<pre>
결과
(50, 10000)
</pre>


```py
import matplotlib.pyplot as plt

def draw_fruits(arr, ratio=1):
    n = len(arr)    # n은 샘플 개수입니다
    # 한 줄에 10개씩 이미지를 그립니다. 샘플 개수를 10으로 나누어 전체 행 개수를 계산합니다. 
    rows = int(np.ceil(n/10))
    # 행이 1개 이면 열 개수는 샘플 개수입니다. 그렇지 않으면 10개입니다.
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, 
                            figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:    # n 개까지만 그립니다.
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()
```

```py
draw_fruits(pca.components_.reshape(-1, 100, 100))
```
<pre> 
결과
<img width="647" alt="image" src="https://github.com/piolink/pyree/assets/75467180/1e5dc2a0-b627-44b2-855a-0d6f28d8e4a9">
</pre>


### 원본 데이터 재구성
10,000개의 특성을 50개로 줄였습니다. 이는 어느 정도 데이터 손실을 발생 시킵니다. **PCA 클래스는** 이를 위해 **inverse_transform()** 메서드를 제공합니다. 


```py
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)
```
<pre>
결과
(300, 10000)
</pre>

```py
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])
    print("\n")
```
<pre> 
<img width="627" alt="image" src="https://github.com/piolink/pyree/assets/75467180/b0b09302-5389-4dca-814e-319d7041a233">

<img width="679" alt="image" src="https://github.com/piolink/pyree/assets/75467180/b1cb0fc7-bb4e-4ffe-b074-ad9ed06029e4">

<img width="636" alt="image" src="https://github.com/piolink/pyree/assets/75467180/e581055f-cbcc-45f5-9f97-b6857157f3dc">

</pre>
> 거의 모든 과일이 잘 복원되었습니다. 만약 주성분을 최대로 사용하였다면 완변하게 원본 데이터를 재구성 할 수 있었을 것입니다.

### 설명된 분산
주성분이 원본 데이터의 분산을 얼마나 잘 나타내는지 기록한 값을 <span style="color:violet">설명된 분산 </span> 이라고 합니다. 


```py
print(np.sum(pca.explained_variance_ratio_))
```
<pre>
결과
0.9215967336068237
</pre>



```py
plt.plot(pca.explained_variance_ratio_)
plt.show()
```
<pre>
결과
<img width="424" alt="image" src="https://github.com/piolink/pyree/assets/75467180/c8c2725e-becb-40a2-a15e-64d88a192c2a">
</pre>


### 다른 알고리즘과 함께 사용하기
사이깃런의 LogisticRegression 모델을 만듭니다.

```py
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
```

```py
target = np.array([0] * 100 + [1] * 100 + [2] * 100)
```
> cross_validate()로 교차 검증 수행


```py
from sklearn.model_selection import cross_validate

scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))
```
<pre>
결과

0.9966666666666667
2.2165074825286863
</pre>

```py
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))
```

<pre>
결과

1.0
0.030451583862304687
</pre>

> n_components 매개변수에 주성분의 개수를 지정했습니다. 


```py
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)

print(pca.n_components_)
```
> 0 ~ 1 사이의 비율을 실수로 입력

<pre>
결과

2
</pre>


k 평균 알고리즘으로 클러스터를 찾아보겠습니다.

```py
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts=True))
```
<pre>
결과

(array([0, 1, 2], dtype=int32), array([110,  99,  91]))
</pre>


KMeans가 찾은 레이블을 사용해 과일 이미지를 출력하겠습니다.

```py
for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")
```
<pre>
결과
<img width="641" alt="image" src="https://github.com/piolink/pyree/assets/75467180/fb8c9685-1bac-41d1-b8dd-4ccf534b350e">

<img width="639" alt="image" src="https://github.com/piolink/pyree/assets/75467180/4763c795-9d04-49cc-a35d-295fbbb5b328">

<img width="632" alt="image" src="https://github.com/piolink/pyree/assets/75467180/ffb2962c-46df-4145-a6fd-1a36680d689f">

</pre>


km.lables_를 사용해서 클러스터 별로 나누어 산점도를 그려봅시다

```py
for label in range(0, 3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:,0], data[:,1])
plt.legend(['apple', 'banana', 'pineapple'])
plt.show()
```

<pre>
결과
<img width="497" alt="image" src="https://github.com/piolink/pyree/assets/75467180/f04e5c82-1029-434a-a981-85587e77385f">

</pre>
> 사과와 파인애플의 경계가 가깝게 붙어있습니다.


  <hr/>

* 차원축소는 원본데이터의 특성을 적은 수의 새로운 특성으로 변환하는 비지도 학습의 한 종류입니다.

* 주성분 분석은 차원 축소 알고리즘의 하나로 데이터에서 가장 분산이 큰 방향을 찾는 방법입니다.

* 설명된 분산은 주성분 분석에서 주성분이 얼마나 원본 데이터의 분산을 잘 나타내는지 기록한 것입니다. 이와 같은 경우에는 사이킷런 PCA 클래스는 주성분 설명된 분산의 비율을 지정하여 주성분 분석을 수행할 수 있습니다. 
