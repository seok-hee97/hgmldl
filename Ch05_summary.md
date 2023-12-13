# 5. 트리 알고리즘
## [5-1]결정 트리(Decision Tree)

- 핵심 키워드
  - 결정 트리
  - 불순도
  - 정보 이득
  - 가지치기
  - 특성 중요도

### 로지스틱 회귀로 와인 분류하기


```python
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')
```

#head() 메서드를 통해 wine 변수에 데이터프레임이 잘 들어갔는지 확인  
#sample() 메서드 추천(위에서가 아니라 샘플링(랜덤픽)해서 데이터 프레임 파악


```python
wine.head()     #default 5
# wine.sample()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alcohol</th>
      <th>sugar</th>
      <th>pH</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4068</th>
      <td>9.5</td>
      <td>1.4</td>
      <td>3.21</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#info() 데이터프레임의 각 열의 데이터 타입과 누락된 데이터 확인하는 유용


```python
wine.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6497 entries, 0 to 6496
    Data columns (total 4 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   alcohol  6497 non-null   float64
     1   sugar    6497 non-null   float64
     2   pH       6497 non-null   float64
     3   class    6497 non-null   float64
    dtypes: float64(4)
    memory usage: 203.2 KB


#describe() 열에 대한 간략한 통계를 출력  
(mean(평균),std(표준편차), min(최소), ,max(최대) +중간값(50%), 1사분위수(25%), 3사분위수(75%))


```python
wine.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alcohol</th>
      <th>sugar</th>
      <th>pH</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>10.491801</td>
      <td>5.443235</td>
      <td>3.218501</td>
      <td>0.753886</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.192712</td>
      <td>4.757804</td>
      <td>0.160787</td>
      <td>0.430779</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.000000</td>
      <td>0.600000</td>
      <td>2.720000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.500000</td>
      <td>1.800000</td>
      <td>3.110000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10.300000</td>
      <td>3.000000</td>
      <td>3.210000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>11.300000</td>
      <td>8.100000</td>
      <td>3.320000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.900000</td>
      <td>65.800000</td>
      <td>4.010000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



#데이터프레임 -> numpy 배열  
#wine 데이터프레임에서 처름 3개의 열을 넘파이 배열로 바꿔서 data 배열에 저장  
#class 열은 넘파이 배열로 바꿔서 target 배열에 저장


```python
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
```

#데이터 세트 -> 훈련세트와 테스트 세트로 split!!  
#[parameter]  
#test_size =0.2(train_size= 0.8)        default:0.25  
#radom_state= 42        #radom seed 값 (시드값을 정하면 알면 같은 랜덤값을 구할 수 있다)


```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)
```

#shape() 메서드 사용해서 train_set / test_set 확인


```python
print(train_input.shape, test_input.shape)
```

    (5197, 3) (1300, 3)


#StandardScaler 클래스를 사용해서  
#훈련세트, 테스트세트를 전처리


```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

#표준점수로 변환된 train_scaled와 test_scaled를 사용해  
#로지스틱 회귀모델 훈련


```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
```

    0.7808350971714451
    0.7776923076923077


### 설명하기 쉬운 모델과 어려운 모델

#lr.coef_ : 로지스틱 회귀가 학습한 계수  
#lr.intercept_ : 로지스틱 회귀가 학습한 절편


```python
print(lr.coef_, lr.intercept_)
```

    [[ 0.51270274  1.6733911  -0.68767781]] [1.81777902]


### 결정 트리(Decision Tree)

> like 스무고개  
> 데이터를 잘 나눌 수 있는 질문을 찾는다면 계속 질분을 추가해서 분류 정확도를 높일 수 있음


```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))     #훈련 세트
print(dt.score(test_scaled, test_target))       #테스트 세트
```

    0.996921300750433
    0.8592307692307692


#Result : 훈련세트에 비해 테스트 세트의 성능이 낮음 -> 과대적합(Overfit)

#위에서 학습한 모델을 사이킷런의 plot_tree()와 matplot() 사용해서 시각화


```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()
```


    
![5-1_26_0](https://github.com/piolink/pyree/assets/98581131/1140b613-921c-4d9d-8be6-fe64c2d2766e)
    


plot_tree() 함수에서 트리의 깊이를 제한해서 출력  
max_depth : tree 최대 깊이  
filled  :클래스에 맞게 노드의 색을 칠할 수 있음  
feature_names : 특성의 이름을 전달  


```python
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```


    
![5-1_28_0](https://github.com/piolink/pyree/assets/98581131/9e005df3-b5f9-44f7-a86a-4f03f0eefd10)
    


### 불순도(impurity)
: 트리 모델에서 데이터를 분류하는 기준을 정하는것
DecisionTreeClassifier 클래스의 criterion 매개변수의 기본값 'gini'

- 지니 불순도
$${ 지니 불순도 = 1 - (음성 클래스 비율^2 + 양성 클래스 비율^2)}$$

- 노드의 클래스의 비율이 정확히 1/2씩 있다면 지니 불순도 0.5 최악       ->제대로 분류되지 못한 거
- 노드에 하나의 클래스만 있다면 지니의 불순도는 0 <- 순수 노드

##### 결정트리 모델은 부모 노드(parent node)와 자식 노드(child node) 불순도 차이가 가능한 크도록 트리를 생성시킴

{부모의 불순도 - (왼쪽 노드의 샘플 수 / 부모의 샘플 수 ) * 왼쪽 노드 불순도 - (오른쪽 노드 샘플 수 / 부모의 샘플 수) * 오른쪽 노드 불순도}

> #부모와 자식 노드 사이의 불순도 차이를 정보 이득(information gain)이라 함

- 엔트로피 불순도  
criterion = 'entropy'  
{엔트로피 불순도 = -(음성 클래스 비율) * log2(음성 클래스 비율) - 양성 클래스 비율 * log2(양성 클래스 비율)}

> 보통 기본값인 지니 불순도와 엔트로피 불순도의 결과차이 크지 않음

### 가지치기

가지치기를 하는 가장 간단한 방법은 트리의 최대 깊이 지정  
max_depth = 3 으로 최대 깊이 지정


```python
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
```

    0.8454877814123533
    0.8415384615384616



```python
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```


    
![5-1_33_0](https://github.com/piolink/pyree/assets/98581131/170ea313-6ae0-4a60-a94c-1b29174534e3)
    



```python
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)

print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))
```

    0.8454877814123533
    0.8415384615384616



```python
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```


    
![5-1_35_0](https://github.com/piolink/pyree/assets/98581131/d92c88ad-2fcf-485e-b96f-0d05d0e1fd41)
    


특성 중요도는 결정 트리 모델의 feature_importances_ 속성에 저장


```python
print(dt.feature_importances_)
```

    [0.12345626 0.86862934 0.0079144 ]


##### [확인문제]


```python
dt = DecisionTreeClassifier(min_impurity_decrease=0.0005, random_state=42)
dt.fit(train_input, train_target)

print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))
```

    0.8874350586877044
    0.8615384615384616



```python
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```


    
![5-1_40_0](https://github.com/piolink/pyree/assets/98581131/dd6cb23a-61fc-41f7-bd6c-31eca72ba019)
    

## [5-2]교차 검증과 그리드 서치

- 핵심 키워드
  - 검증 세트
  - 교차 검증
  - 그리드 서치
  - 랜덤 서치

## 검증 세트(validation set)
: 테스트 세트를 사용하지 않으면 과대적합/과소적합 판단 어려움.  
-> 훈련세트를 또 나눠서 검증 세트(validation set)

#보통 20-30% 테스트와 검증 세트로 떼어놓음


```python
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')
```


```python
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
```

train_test_split() 이용해서 훈련 세트와 테스트 세트 split!!


```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)
```

train_input과 train_target을 이용해서 다시.  
sub_input, sub_target과 검증세트 val_input, val_target 만듬


```python
sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)
```


```python
print(sub_input.shape, val_input.shape)
```

    (4157, 3) (1040, 3)



```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)

print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))
```

    0.9971133028626413
    0.864423076923077


## 교차 검증(cross validation)
> 교차 검증을 이용하면 안정적인 검증 점수를 얻고 훈련에 더 많은 데이터 사용 가능


#훈련 세트를 세부분으로 나눠서 수행하는 것 -> 3-폴드 교차 검증  
통칭 k-폴드 교차 검증(k-fold cross validation)


#보통 5-fold cross validation or 10-fold cross validation 많이 사용

사이킷런에 cross_validate() 함수 존재  
(전신인 cross_val_score() 함수도 존재)

fit_time, score_time, test_score 키를 가진 딕셔너리 반환

cross_validate(cv=5) default : 5-fold cross validation


```python
from sklearn.model_selection import cross_validate

scores = cross_validate(dt, train_input, train_target)
print(scores)
```

    {'fit_time': array([0.00768113, 0.006881  , 0.00670505, 0.00590205, 0.00520992]), 'score_time': array([0.00054193, 0.00055909, 0.00047588, 0.00047112, 0.00043583]), 'test_score': array([0.86923077, 0.84615385, 0.87680462, 0.84889317, 0.83541867])}



```python
import numpy as np

print(np.mean(scores['test_score']))
```

    0.855300214703487


주의점: cross_validate() 훈련 세트를  섞어 폴드를 나누지 않음   
앞서 train_test_split() 함수로  전제 데이터를 섞은 후 훈련 세트를 준비했기 대문에 따로 섞을 필요 없었음

만약 교차 검증 할때 훈련 세트를 섞으러면 분할기(splitter) 지정

타깃 클래스를 골고루 나누기 위해 StatifiedKFold 사용


```python
from sklearn.model_selection import StratifiedKFold
scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))
```

    0.855300214703487



```python
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))
```

    0.8574181117533719


## 하이퍼파라미터 튜닝
: 사용자가 지정해야만 하는 파라미터를 하이퍼파라미터

>사람의 개입 없이 하이퍼파라미터 튜닝을 자동으로 수행하는 기술 'AutoML'이라고 부름

사이킷런에서 제공하는 그리드 서치(Grid Search)

GridSearchCV 클래스 하이퍼파라미터 탐색과 교차 검증 한번에 수행


```python
from sklearn.model_selection import GridSearchCV
#Make var params
params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
```


```python
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)

#njobs CPU 병렬 실행 여부       default:1   사용 x
```


```python
gs.fit(train_input, train_target)
```



가장 높은 점수의 모델(같은 매개변수로 다시 학습)  
best_estimator_ 속성에 저장


```python
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
```

    0.9615162593804117


최적의 매개변수 best_params_ 속성에 저장


```python
print(gs.best_params_)
```

    {'min_impurity_decrease': 0.0001}


각 매개변수에서 수행한 교차 검증의 평균 점수는 cv_result_ 속성의 'mean_test_score' 키에 저장


```python
print(gs.cv_results_['mean_test_score'])
```

    [0.86819297 0.86453617 0.86492226 0.86780891 0.86761605]


numpy의 argmax() 함수 사용하면 가장 큰 값의 인덱스 값 리턴


```python
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])
```

    {'min_impurity_decrease': 0.0001}


numpy 모듈의 arange 함수는 반열린구간 [start, stop) 에서 step 의 크기만큼.  
일정하게 떨어져 있는 숫자들을 array 형태로 반환해 주는 함수


```python
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)
          }
```


```python
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
```






```python
print(gs.best_params_)
```

    {'max_depth': 14, 'min_impurity_decrease': 0.0004, 'min_samples_split': 12}



```python
print(np.max(gs.cv_results_['mean_test_score']))
```

    0.8683865773302731


### 랜덤 서치(Random Search)
: 랜덤 서치는 매개변수 값의 목록을 전달하는 것이 아니라 매개변수를 샘플링할 수 있는 확률 분포 객체를 전달


```python
#수학 라이브러리
from scipy.stats import uniform, randint
```


```python
rgen = randint(0, 10)
rgen.rvs(10)
```




    array([7, 6, 5, 4, 8, 4, 5, 5, 0, 1])




```python
np.unique(rgen.rvs(1000), return_counts=True)
```




    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
     array([107, 103,  75,  87, 114, 106, 101,  99, 105, 103]))




```python
ugen = uniform(0, 1)
ugen.rvs(10)
```




    array([0.95712558, 0.95636048, 0.4406807 , 0.82250232, 0.75499527,
           0.75727956, 0.19566117, 0.52756242, 0.52128849, 0.72410768])




```python
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25),
          }
```


```python
from sklearn.model_selection import RandomizedSearchCV

gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, 
                        n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)
```






```python
print(gs.best_params_)
```

    {'max_depth': 39, 'min_impurity_decrease': 0.00034102546602601173, 'min_samples_leaf': 7, 'min_samples_split': 13}



```python
print(np.max(gs.cv_results_['mean_test_score']))
```

    0.8695428296438884



```python
dt = gs.best_estimator_

print(dt.score(test_input, test_target))
```

    0.86

# [5-3]트리의 앙상블

- 핵심 키워드
  - 앙상블 학습
  - 랜덤 포레스트
  - 엑스트라 트리
  - 그라디언트 부스팅

#### 정형 데이터와 비정형 데이터

- 정형 데이터(structed data)
: 가공된 데이터 ex] csv, excel, database

- 비정형 데이터(unstructed data)
: Ex] 사진, 영상 .. 등


정형 데이터를 다루는 데 가장 뛰어난 성과를 내는 알고릐즘이  
 앙상블(ensemble learning) (결정 트리 기반으로 만들어짐)

## 랜덤포레스트(Random Forest)
- 앙상블 학습의 대표 알고리즘
- 안정적인 성능
- 결정트리르 랜덤하게 만들어 숲을 만듬(각 결정트리의 예측을 사용해 최종 예측)

![Screenshot 2023-07-17 at 11 32 42 PM](https://github.com/piolink/pyree/assets/98581131/18efeaca-ae32-47e3-ac16-dbd0ab8a711f)


#### 렌담 포레트스 훈련 방법

![Screenshot 2023-07-17 at 11 33 10 PM](https://github.com/piolink/pyree/assets/98581131/1e32213a-958e-4b01-9328-c1fe62843d5a)

부트스트랩 사용
: 데이터 세트에서 중복을 허용하여 데이터 샘플링하는 방식

[예시]
사이킷런의 RandomForestClassfier 와인 분류 모델에 적용


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
```


```python
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
#retain_train_score =True : 검증 점수뿐만 아니라 훈련 세트에 대한 점수도 같이 반환          #default =False
#n_jobs=-1: cpu 병렬로 사용

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

    0.9973541965122431 0.8905151032797809


> 랜덤 포레스틑 결정 트리의 앙상블이기 때문에 DecisionTreeClassifier가 제공하는 중요한 매개 변수 모두 제공


```python
rf.fit(train_input, train_target)
print(rf.feature_importances_)
```

    [0.23167441 0.50039841 0.26792718]



```python
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)

rf.fit(train_input, train_target)
print(rf.oob_score_)
```

    0.8934000384837406


## 엑스트라트리(Extra Trees)
- 랜덤 포레스트와 비슷하게 동작
- 전체 특성 중 일부 특성을 랜덤하게 선택하여 노드 분할에 사용
- 랜더 포레스트와 차이점은 부트스트랩 샘플 사용 x
- 즉각 결정 트리를 만들 때 전체 훈련 세트를 사용

사이킷런에서 제공하는 ExtraTressClassifier 사용


```python
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

    0.9974503966084433 0.8887848893166506


> 보통 엑스트라 트리가 무작위성이 좀 더 크기 때문에 랜덤 포레스트보다 더 많은 결정 트리 훈련 필요  
> 하지만 랜덤하게 노드를 분할하기 때문에 빠른 계산 속도가 장점


```python
et.fit(train_input, train_target)
print(et.feature_importances_)
```

    [0.20183568 0.52242907 0.27573525]


## 그레이디언트 부스팅(Gradient boosting)
- 깊이가 얕은 결정 트리를 사용하여 이전 트리의 오차를 보완하는 방식으로 앙상블 하는 방법
- 그레이디언트 부스팅은 결정 트리르 계속 추가하면서 가장 낮은 곳을 찾아 이동(4장  경사 하강법과 같은 메커니즘)
- 결정 트리의 개수를 늘려도 과대적합에 매우 강함
- 학습률을 증가시키고 트리의 개수를 늘리면 성능 향상

GradientBoostingClassifier 기본적으로 깊이 3인 결정 트리 100개 사용


```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

    0.8881086892152563 0.8720430147331015



```python
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

    0.9464595437171814 0.8780082549788999



```python
gb.fit(train_input, train_target)
print(gb.feature_importances_)
```

    [0.15872278 0.68010884 0.16116839]


## 히스토그램 기반 부스팅(Histogram-based Gradient Boosting)
: 먼저 특성을 256개 구간으로 나눔 -> 노드를 분할할 때 최적의 분할을 빠르게 찾을 수 있음

- 256개의 구간 중에서 하나를 떼어놓고 누락된 값을 위해서 사용 -> 입력에 누락된 특성이 있더라도 전처리 필요 X

트리의 개수를 지정하는데 max_estimators 대신 부스팅 반복 횟수를 지정하는 max_iter 사용


```python
# 사이킷런 1.0 버전 아래에서는 다음 라인의 주석을 해제하고 실행하세요.

from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

    0.9321723946453317 0.8801241948619236



```python
from sklearn.inspection import permutation_importance

hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target, n_repeats=10,
                                random_state=42, n_jobs=-1)
print(result.importances_mean)
```

    [0.08876275 0.23438522 0.08027708]



```python
result = permutation_importance(hgb, test_input, test_target, n_repeats=10,
                                random_state=42, n_jobs=-1)
print(result.importances_mean)
```

    [0.05969231 0.20238462 0.049     ]



```python
hgb.score(test_input, test_target)
```




    0.8723076923076923



#### XGBoost


```python
from xgboost import XGBClassifier

#tree_method = 'hist' 사용하면 히스토르램 기반 그레디언트 부스팅 알고리즘 사용 가능
xgb = XGBClassifier(tree_method='hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

    0.9555033709953124 0.8799326275264677


#### LightGBM
- 히스토그램 기반 그레디언트 부스팅 라이브러리 made by MS
- 속도가 빠르고 최신 기술 많이 적용


```python
from lightgbm import LGBMClassifier

lgb = LGBMClassifier(random_state=42)
scores = cross_validate(lgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

    0.935828414851749 0.8801251203079884

