# Python 조건문, 반복문

# 조건문 - if
# Python 은 C, Java 등과 같이 코딩블럭을 { }로 나타내지 않고 들여쓰기를 사용.
# 동일한 블럭의 들여쓰기는 모두 동일한 수의 공백을 사용해야 함.

a = 1

# 보통의 if문
if a > 0:
    print("a == ", a)
    print("positive number")
elif a == 0:
    print("a == ", a)
    print("zero")
else:
    print("a == ", a)
    print("negative number")

# list, dict 등을 인자로 받는 if 문
list_data = [10, 20, 30, 40, 50]
dict_data = {'key1': 1, 'key2': 2}

if 45 in list_data:
    print("45 is in list_data")
else:
    print("45 is not in list_data")

if 'key1' in dict_data:
    print("key1 is in dict_data")
else:
    print("key1 is not in dict_data")


#######################################################################

# 반복문 - for
# range 를 사용하는 방법과 list, dict 등을 사용하는 방법도 있다.

# range() 함수
# 시작값 ~ (마지막 값 -1)
for data in range(10):
    print(data, end=" ")
print()
for data in range(0, 10):
    print(data, end=" ")
print()
for data in range(0, 10, 2):
    print(data, end=" ")
print()

# list, dict 등을 in 뒤에 쓴다.
list_data = [10, 20, 30, 40, 50]

for data in list_data:
    print(data, end=" ")
print()

for data in dict_data:
    print(data, end=" ")
print()

for key, value in dict_data.items():
    print(key, value)

#######################################################################

# 반복문 - list comprehension
# 리스트의 [...] 괄호 안에 for 루프를 사용하여 반복적으로 표현식을 실행해서 리스트 요소들을 정의하는 방법.
# 머신러닝 코드에서 자주 사용되는 기법

list_data = [x**2 for x in range(5)]
print(list_data)

raw_data = [ [1, 10], [2, 15], [3, 30], [4, 55] ]

all_data = [x for x in raw_data]
x_data = [x[0] for x in raw_data]
y_data = [x[1] for x in raw_data]

print("all_data == ", all_data)
print("x_data == ", x_data)
print("y_data == ", y_data)

# list comprehension 으로 짝수만 출력하는 코드.
even_number = [x for x in range(10) if x % 2 == 0]
print(even_number)

#######################################################################

# 반복문 - while

data = 5

while data >= 0:
    print("data == ", data)
    data -= 1
    if data == 2:
        print("break here")
    else:
        print("continue here")
        continue

#######################################################################

# ******
for _ in range(6):
    print("*", end="")
print('\n')

# *****
# ****
# ***
# **
# *

for i in range(5):
    print("{}".format("*") * (5 - i))

apart = [ [101, 102, 103, 104], [201, 202, 203, 204], [301, 302, 303, 304], [401, 402, 403, 404]]
arrears = [101, 203, 301, 404]

# 중첩 루프를 이용해 신문 배달을 하는 프로그램을 작성시오.
# 단, 아래에서 arrears 리스트는 신문 구독료가 미납된 세대에 대한 정보를 포함하고 있는데,
# 해당 세대에는 신문을 배달하지 않아야 함
for index, generation in enumerate(apart):
    for i in arrears:
        if i in generation:
            apart[index].pop(generation.index(i))

for data in apart:
    print(data)
