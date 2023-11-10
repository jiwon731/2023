import math

# 데이터 리스트 (양수와 음수 포함)
data = [1, -2, 3, -4, 5]

# 결과를 저장할 리스트
result = []

# 데이터에 대한 루트 연산 수행
for value in data:
    if value >= 0:
        result.append(math.sqrt(value))  # 양수의 경우 루트 연산
    else:
        result.append(-math.sqrt(abs(value)))  # 음수의 경우 절댓값 취하고 루트 연산 후 다시 음수 부호 적용

# 결과 출력
print(result)