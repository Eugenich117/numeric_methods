import matplotlib.pyplot as plt

x = 0
h = 0.5
y = 0
f = 1
X1 = [x]
Y1 = [y]
while x < 1:
    x += h
    f = x / (x + 1)
    y = f
    print('x =', x, 'y =', y)
    X1.append(x)
    Y1.append(y)
x = 0
h = 0.25
y = 0
f = 1
X2 = [x]
Y2 = [y]
while x < 1:
    x += h
    f = x / (x + 1)
    y = f
    print('x =', x, 'y =', y)
    X2.append(x)
    Y2.append(y)
x = 0
h = 0.2
y = 0
f = 1
X3 = [x]
Y3 = [y]
while x < 1:
    x += h
    f = x / (x + 1)
    y = f
    print('x =', x, 'y =', y)
    X3.append(x)
    Y3.append(y)
x = 0
h = 0.1
y = 0
f = 1
X4 = [x]
Y4 = [y]
while x < 1:
    x += h
    f = x / (x + 1)
    y = f
    print('x =', x, 'y =', y)
    X4.append(x)
    Y4.append(y)


plt.plot(X1, Y1, '-', X2, Y2, '-', X3, Y3, '-', X4, Y4, '-')
plt.show()