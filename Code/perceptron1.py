import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

x1 = [1,1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 3, 2, 1, 2, 1, 2, 5]
y1 = [2,1, 1, 3, 1, 1, 0.4, 0.5, 0.4, 2, 1, 4, 4,0,2,4,0, 2, 2]

x2 = [ 4, 5, 6, 7, 8, 10, 6]
y2 = [ 6, 7, 6, 5, 4, 3, 4]


# plt.plot(x1, y1, 'bo')
# plt.plot(x2, y2, 'ro')

# plt.show()


# -------------
x = []
y = []

for i in range(len(x1)):
	x.append((x1[i], y1[i]))
	y.append(0)

for i in range(len(x2)):
	x.append((x2[i], y2[i]))
	y.append(1)

print(x)
print(y)


clf = Perceptron(tol=1e-3, verbose=True)

clf.fit(x, y)

print(clf.score(x, y))


print(clf.get_params())

for i in range(0, len(x)):
	plt.plot(x[i][0], x[i][1], 'o')
plt.show()

for i in range(0, len(x)):
	if y[i] == 0:
		plt.plot(x[i][0], x[i][1], 'bo')
	else:
		plt.plot(x[i][0], x[i][1], 'ro')


plt.show()



x3 = 7
y3 = 7

print(clf.predict([(x3, y3)]))

if y[i] == 1:
	print('rosu')
else:
	print('albastru')


