
cluch = [0.18, 0.17, 0.16, 0.15, 0.14, 0.13]
income = 0
total_income = 0
cash = 1_900_000
print('Проценты по вкладу уходят в карман')
for i in range(6):
    income = ((cash * cluch[i]) / 12 ) * 2
    #cash += income
    total_income += income
    print(f"сумма на вкладе {cash:.1f}, доход {income:.1f}")
print(f"доход {income}, сумма на вкладе {cash} ")

print('Проценты по вкладу реинвестируются')
for i in range(6):
    income = ((cash * cluch[i]) / 12 ) * 2
    cash += income
    total_income += income
    print(f"сумма на вкладе {cash:.1f}, доход {income:.1f}")
print(f"доход {income}, сумма на вкладе {cash} ")
