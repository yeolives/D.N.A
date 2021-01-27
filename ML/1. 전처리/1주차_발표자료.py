2020Y-05M-14D 수업자료 

0. index 이해

a = [1,2,3,4,5,6,7,8]
len(a) # 8

a[0] 
a[7] # 8-1
a[-1]
a[3] - a[3-1]

a[:] # 전체
a[2:] # 인덱스[2] = 3 부터 전체
a[:2] # 인덱스[2] 까지  
a[:-1] # 마지막은 빼고
a[1:-1] # 처음과 마지막만 빼고

a[::] # 전체
a[0:3:] # == a[0:3]
a[0::2] # [0]인덱스부터시작해서 2씩더해가면서 출력
a[::-1] # 역순정렬
a[-1::] # 마지막값만 출력


# 행렬접근법
from IPython.display import Image
Image("./Matrix.png") # 행렬이론


a = [[1,2,3],
     [4,5,6]]


len(a) # 열의개수 == 2
len(a[0]) # 행의개수 == 3

a[0] # 첫번째 행  
a[1] # 2번째 행

a[0][0] # 1번째 행 1번째 열의 값
a[1][0] # 2번째 행 1번째 열의 값

a[::-1] # 두개의 행 위치 바꾸기


for i in range(len(a)): # 2 ( 0, 1 )
    for j in range(len(a[i])): # 3 (0, 1, 2)
        print(a[i][j]) # 원소에 접근하는 법 






1. Numpy(Numerical Python)의 이해

list와는 비슷하지만 배열의 규모가 커질수록 데이터 저장 및 처리에 훨씬 더 효율적이다.
데이터 분석을 포함해 수학과 과학연산을 위한 파이썬 기본 패키지라고 한다

왜쓰는가? 
NumPy는 python에서 수학/과학 연산을 위한 다차원 배열 객체를 지원한다.
NumPy의 다차원 배열 자료형인 ndarray는 scipy, pandas 등 다양한 파이썬 패키지의 기본 자료형으로 사용되기 때문이다.
또한 For 문과 같이 반복적인 연산 작업을 배열단위로 처리하여, 효율적인 코딩이 가능하다.

import numpy as np # 패키지불러오기

#Numpy vs List
A = [[1,0],
     [0,1]]
B = [[1,1], 
     [1,1]]
A+B 

C = np.array([[1,0],
              [0,1]])
D = np.array([[1,1],
              [1,1]])
C+D 

list는 A + B 시 행렬 연산이 안된다.(리스트 연산이 되어서 리스트에 값이 추가됨)
list로 행렬 연산을 하려면 for문을 돌려서 각 각 연산을 해줘야한다.

A[0][0]
C[0,0]

A[:][0]
C[:,0]
list와 array는 인덱스로 접근하는 방식이 다르다. 


# array형성 (vector)
A = np.array([1,2,3])
B = np.array([4,5,6])
A

# vector A, B 형상 출력 => shape
print("A.shape ==", A.shape, ", B.shape ==", B.shape)

# vector A, B 차원 출력 => ndim
print("A.ndim==", A.ndim, "B.ndim ==", B.ndim)

C = A.reshape(1,3)
print(C.shape)
print(C.ndim)

list와 달리 대괄호로 묶지 않고 괄호로 묶는다.
대괄호 안에 대괄호가 있는 건 행렬, 숫자만 있으면 vector
(3,) 는 1행에 3열이나 3행에 1열이라는 뜻이다. 또한 vector로 인식된다.
모호하므로 reshape로 정확하게 잡아준다.
reshape를 해서 행렬로 인식되어서 차원이 1에서 2로 변경된다.
A.shape, A.ndim 숙지할 것
행렬의 사칙연산을 할 때 shape이 같아야 한다

# 행렬생성
A = np.array([[1,2,3],[4,5,6]])
B = np.array([[-1,-2,-3],[-4,-5,-6]])

# matrix A, B형상 출력 => shape
print("A.shape ==", A.shape, "B.shape ==", B.shape)
# matrix A, B차원 출력
print("A.ndim ==", A.ndim, "B.ndim ==", B.ndim)


## Numpy broadcast
A = np.array([[1,2],[3,4]])
b = 5
print(A+b)

C = np.array([[1,2],[3,4]])
D = np.array([4,5])
print(C+D)

행렬 사칙연산 시 크기가 같아야하는데 broadcast로 행렬의 크기를 자동으로 변경시켜 계산한다.
차원이 작은 쪽이 큰쪽의 행 단위로 반복적으로 숫자를 끼워 크기를 맞춤.
행렬곱(dot product)일 땐 사용 안됨.

np.zeros((2,3))

X = np.array([[2,4,6], 
              [1,2,3], 
              [0,5,8]])  
print("np.max(X) ==", np.max(X, axis=0))  
print("np.min(X) ==", np.min(X, axis=0))  
print("np.max(X) ==", np.max(X, axis=1))  
print("np.min(X) ==", np.min(X, axis=1))


axis 1은 기준이 행을 기준으로 최대값,최소값을 빼낸다. 0은 열을 기준으로 빼낸다.

np.log(X)
np.exp(X)
np.argmin(X)
np.argmax(X)
np.mean(X)
np.std(X)






2. Pandas의 이해

Pandas는 파이썬에서 사용하는 데이터분석 라이브러리로, 
행과 열로 이루어진 데이터 객체를 만들어 다룰 수 있게 되며 
보다 안정적으로 대용량의 데이터들을 처리하는데 매우 편리한 도구 이다.
import pandas as pd

※Pandas 자료구조
Pandas에서는 기본적으로 정의되는 자료구조인 Series와 Data Frame을 사용한다.
이 자료구조들은 빅 데이터 분석에 있어서 높은 수준의 성능을 보여준다.


2 - 1 Series살펴보기
from IPython.display import Image
Image("./series.png") # Series 형상

Park = pd.Series([92600, 92400, 92100, 94300, 92300])
Park

Series 객체는 파이썬 리스트와 달리 인덱싱 값을 지정할 수 있다.

Park2 = pd.Series([92600, 92400, 92100, 94300, 92300], index=['2016-02-19',
                                                            '2016-02-18',
                                                            '2016-02-17',
                                                            '2016-02-16',
                                                            '2016-02-15'])
Park2


## series 연산
Hyun = pd.Series([10, 20, 30], index=['naver', 'sk', 'kt'])
Yong = pd.Series([10, 30, 20], index=['kt', 'naver', 'sk'])

Hyun + Yong
Hyun * Yong
같은 인덱스를 갖는 값끼리 정상적으로 덧셈이 수행된 것을 확인할 수 있다.




2 - 2 DataFrame살펴보기

raw_data = {'col0': [1, 2, 3, 4],
            'col1': [10, 20, 30, 40],
            'col2': [100, 200, 300, 400]}

data = pd.DataFrame(raw_data)
print(data)

type(data['col0'])

type의 리턴 값을 살펴보면 DataFrame에 있는 각 칼럼은 Series 객체임을 알 수 있다. 
즉, DataFrame을 인덱스가 같은 여러개의 Series 객체로 구성된 자료구조로 생각해도 좋다.

Image('./dataframe.png')

Image('./dataframe2.png')


pd.DataFrame(data = np.array([[1,2,3],[4,5,6],[7,8,9]]), columns = ['A', 'B', 'C'])


## 인덱스와 칼럼 접근
PHY = pd.DataFrame(data = np.array([[1,2,3],[4,5,6],[7,8,9]]), columns = ['A', 'B', 'C'])

PHY['A']
PHY.A

PHY.index
PHY.index[0]

PHY.columns
PHY.columns.tolist()


PHY['A'][0]
PHY['A'][0:2]

PHY[['A','B']][0:2]

PHY['C'] = [0,0,0]
PHY.C

## 행,열 추출
df.iloc[[행],[열]] # Data의 행 번호 활용, integer(정수)만 가능 
df.loc[[행],[열]] # DataFrame index 활용, 아무 것이나 활용 가능

- loc은 인덱스 기준으로 행 데이터 읽기
- iloc은 행 번호를 기준으로 행 데이터 읽기 
Image('./colind.png')

PHY
PHY.loc[0]
PHY.loc[1]

PHY.loc[0,:] # 0번째 인덱스에 대하여 모든열
PHY.loc[0,'A']
PHY.loc[[0,2],'A'] # 0번째와 2번째 인덱스에 대하여 A열
PHY.loc[PHY.index[0],'A'] # 인덱스가 0번째인것에 대하여 'A'열의 값
PHY.loc[PHY.index[0],['A','B']] # 인덱스가 0번째인것에 대하여 'A','B'열


iloc같은 경우는 행과 열 인덱스에 정수리스트를 전달해줘야 한다.(열 이름을 넣어주면 에러가 난다.)

PHY
PHY.iloc[0]
PHY.iloc[1]

PHY.iloc[:,0] # 모든 행에 대하여 0번('A')열 
PHY.iloc[[0,1],[0,2]] # 0번째, 1번째 행번호에 대하여 0번째('A') 2번째('C') 열

PHY.iloc[PHY.index[0],0] # 인덱스가 0번째인것에 대하여 'A'열의 값
PHY.iloc[PHY.index[0],[0]] # 인덱스가 0번째인것에 대하여 'A'열




## 여러가지 데이터프레임을 합치는 방법
P = pd.DataFrame({"id":[1,2,3,4,5],"price":[1300,2200,3500,1600,4000], 
                  "type":['apple','pear','banana','pear','apple'],
                  "orin_price":[900, 2000, 3000, 1200, 2900]})

H = pd.DataFrame({"id":[1,2,3,4,5],"price":[1300,2200,3500,1600,4000], 
                  "type":['apple','pear','banana','pear','apple'],
                  "orin_price":[900, 2000, 3000, 1200, 2900]})

Y = pd.DataFrame({"id":[1,2,3,4,5],"price":[1000,2000,3100,1900,6000], 
                  "type":['orange','pear','banana','apple','apple'],
                  "state":[3, 2, 3, 5, 4]})

a = pd.concat([P,H,Y],axis=1) # axis=1로 했으니깐 행번호를 기준으로 옆으로 붙임.
a = pd.concat([P,H,Y],axis=0) # axis=0으로 했으니깐 열번호를 기준으로 아래로 붙임.

pd.merge(P,Y, on='id')
pd.merge(P,Y, on='type')
pd.merge(P,Y, on='price')


pd.merge(P,Y, on=['id','type'])


pd.merge(P,Y, on='type', how='outer')
pd.merge(P,Y, on='type', how='inner')

- how = 'outer'는 기준열 'type'의 데이터의 중복 여부와 상관없이 모두 출력한다.
- how = 'inner'은 중복된 데이터의 행만 출력한다. 


pd.merge(P,Y, on='type', how='left')
pd.merge(P,Y, on='type', how='right')

- left는 데이터 P를 기준으로 병합한다.
- right는 데이터 Y를 기준으로 병합한다.



## 행,열 삭제
P2 = Y.copy()
P2

P2.drop('state', axis=1, inplace=False) # inplace=True로 해야 적용됨.

# state열을 삭제한다.
P2.drop(1, axis=0, inplace=False) 
# 인덱스 1번째를 삭제한다. 

del P2['state']
P2


## 결측값 처리하기
Y = pd.DataFrame({"id":[1,2,3,4,5],"price":[1000,2000,3100,1900,6000], 
                  "type":['orange','pear','banana','apple','apple'],
                  "state":[3, 2, 3, 5, 4],
                  "eval":[2.1, 5.9, 8.2, np.nan,9.9]})

Y.isna()
Y.isnull()
Y.isna().sum()

Y.dropna()

Y.fillna(0)

Y.fillna(Y['eval'].mean())

Y.fillna('error')

Y.fillna(method='ffill')
Y.fillna(method='pad')

Y.fillna(method='bfill')
Y.fillna(method='backfill')

Y.loc[Y['eval'].isna(), 'eval'] = 3
Y['eval']


# 집계함수
같은 값을 하나로 묶어 통계 또는 집계 결과를 얻기 위해 사용하는 것이 groupby이다.

Fark = pd.DataFrame({
    'city': ['부산', '부산', '부산', '부산', '서울', '서울', '서울'],
    'fruits': ['apple', 'orange', 'banana', 'banana', 'apple', 'apple', 'banana'],
    'price': [100, 200, 250, 300, 150, 200, 400],
    'quantity': [1, 2, 3, 4, 5, 6, 7] 
})

Fark

Fark.groupby('city').mean()
Fark.groupby(['city', 'fruits']).mean()

groupby를 사용하면 기본으로 그룹 라벨이 index가 된다.
index를 사용하고 싶은 않은 경우에는 as_index=False 를 설정하면 된다.

Fark.groupby(['city', 'fruits'], as_index=False).mean()


Fark.groupby('city').get_group('부산') #그룹 안에 데이터를 확인하고 싶은 경우 사용.

Fark.groupby('city').size() # 각 그룹의 크기를 얻음.
Fark.groupby('city').size()['부산']


- Aggregation -
GroupBy.mean()처럼 그룹별로 결과를 얻는 조작을 Aggregation이라고 부른다.
GroupBy 오브젝트에는 Aggregation에 사용할 수 있는 함수가 있다.
Aggregation를 사용하고 싶은 경우에는 agg()를 사용해서 처리할 수 있다.

Fark.groupby('city').agg(np.mean) # == Fark.groupby('city').mean()

Fark.groupby('city').agg({'price': np.mean, 'quantity': np.sum}) # 가격의 평균과 수량의 합계 동시에.



## 추가내용
Fark.values
Fark['price'].values

Fark['city'].value_counts()

Fark.describe()
Fark.describe().transpose()

type(Fark['price'][0])
Fark['price'].astype('float')
type(Fark['price'].astype('float')[0])

# 데이터 로드
(방법1)
train = pd.read_csv("./train.csv")

(방법2)
import os
os.getcwd()
os.chdir('C:/Users/etotm/Desktop/PLACTICE_STUDY')
os.getcwd()

test = pd.read_csv("./test.csv")
test = pd.read_csv("./test.csv", index_col='Pclass')


# 데이터 출력
test.to_csv('sample1.csv')
test.to_csv('sample2.txt', sep='|')



