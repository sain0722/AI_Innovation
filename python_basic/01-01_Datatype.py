# Python DataType

# list
# list 는 배열(Array)과 비슷한 성질을 갖는다.
# index 는 0 부터 시작하며, 파이썬에서는 마이너스(-) 인덱스를 지원함.
#   - 리스트의 마지막부터 역순으로 값을 참조할 수 있음.
#   - 머신러닝 코드에서 슬라이싱과 함께 자주 사용됨.

a = [10, 20, 30, 40, 50]

print("a[0] == ",  a[0],  "a[2] == ", a[2])
print("a[-1] == ",  a[-1],  "a[-2] == ", a[-2])
print()

# list 각 요소의 데이터타입을 다르게 생성할 수 있음.
# list 안에 또 다른 list 를 포함할 수 있음.

b = [10, 20, "Hello", [True, 3.14]]

print("b[0] == ", b[0], ", b[2] == ", b[2], " , b[3] == ", b[3])
print("b[-1] == ", b[-1], ", b[-2] == ", b[-2], " , b[-4] == ", b[-4])

print("b[3][0] == ", b[3][0], ", b[3][1] == ", b[3][1])
print("b[-1][-1] == ", b[-1][-1], ", b[-1][-2] == ", b[-1][-2])

# 빈 list 생성 후 append method 를 이용하여 데이터 추가
#   - 머신러닝 코드에서 정확도 계산, 손실함수 값 저장하기 위해 사용.
# 콜론(:)을 이용한 슬라이싱 기능이 있음.
#   - 슬라이싱을 이용하면 범위를 지정해 부분 리스트를 얻을 수 있음.
#   - 머신러닝을 위해서 반드시 알아야 하는 기능
# a[0:2] => 인덱스 0부터 (2-1) 까지
# a[1:]  => 인덱스 1부터 끝까지
# a[:3]  => 인덱스 처음부터 (3-1) 까지
# a[:-2] => 인덱스 처음부터 (-2-1) 까지
# a[:]   => 인덱스 처음부터 끝까지
a = [10, 20, 30, 40, 50]

print("a[0:2] == ", a[0:2], ", a[1:] == ", a[1:])
print("a[:3] == ", a[:3], ", a[:-2] == ", a[:-2])
print("a[:] == ", a[:])
print()

#######################################################################

# tuple
# list 와 거의 비슷함.
# 차이점
#   - 리스트는 []으로 둘러싸지만, tuple 은 ()으로 둘러싼다.
#   - 리스트 내의 원소는 변경할 수 있지만, tuple 은 변경할 수 없다. (immutable)

a = (10, 20, 30, 40, 50)

print("a[0] == ",  a[0],  "a[-2] == ", a[-2], ", a[:] == ", a[:])
print("a[0:2] == ",  a[0:2],  "a[1:] == ", a[1:])
print()
# a[0] = 100    # a[0] 값을 100으로 변경하려 하기 때문에 TypeError 발생

#######################################################################

# dictionary
# hash 또는 map 과 구조가 비슷함.
# key 와 value 를 할 쌍으로 해서 데이터를 저장한다.

score = {'KIM': 90, 'LEE': 85, 'JUN': 95}
print("score['KIM'] == ", score['KIM'])
score['HAN'] = 100      # 새 원소 추가

# dictionary 는 입력한 순서대로 데이터가 들어가는 것이 아니므로 주의한다.
print(score)

# key, value 값 확인
print("score key == ", score.keys())
print("score value == ", score.values())
print("score items == ", score.items())
print()

#######################################################################

# string
# 문자열은 '' 또는 ""를 사용해서 생성
# 문자열 내의 각각의 값 또한 문자열로 인식됨.
# 문자열을 분리하여 list 로 반환하는 split() 함수
#   - 머신러닝 코드에서 문자열 데이터 전처리를 하기 위해 자주 사용됨.

a = 'A73,CD'
print(a[1], type(a[1]))     # a[1] 은 숫자 7이 아닌 문자열 7

a = a + ', EFG'             # + 연산자 사용
print(a)

# split() 메서드는 특정 separator 를 기준으로 문자열을 분리하여 list 를 리턴.
b = a.split(',')
print(b)
print()

#######################################################################

# [Other] function
# type(data): 입력 data 의 데이터타입을 알려주는 함수
# len(data): 입력 data 의 길이(요소의 개수)를 알려주는 함수

a = [10, 20, 30, 40, 50]
b = (10, 20, 30, 40, 50)
c = {'KIM': 90, 'LEE': 85}
d = 'SEOUL, KOREA'
e = [ [100, 200], [300, 400], [500, 600]]

print(type(a), type(b), type(c), type(d), type(e))
print(len(a), len(b), len(c), len(d), len(e))