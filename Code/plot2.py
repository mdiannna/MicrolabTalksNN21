import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# x1 = [1,1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 3, 2, 1, 2, 1, 2, 5]
# y1 = [2,1, 1, 3, 1, 1, 0.4, 0.5, 0.4, 2, 1, 4, 4,0,2,4,0, 2, 2]

# x2 = [ 4, 5, 6, 7, 8, 10, 6]
# y2 = [ 6, 7, 6, 5, 4, 3, 4]


# plt.plot(x1, y1, 'ro')
# plt.plot(x2, y2, 'bo')

# plt.show()


# # -------------
# x = []
# y = []

# for i in range(len(x1)):
# 	x.append((x1[i], y1[i]))
# 	y.append(0)

# for i in range(len(x2)):
# 	x.append((x2[i], y2[i]))
# 	y.append(1)

# print(x)
# print(y)


# clf = Perceptron(tol=1e-3, verbose=True)

# clf.fit(x, y)

# print(clf.score(x, y))


# print(clf.get_params())

# for i in range(0, len(x)):
# 	plt.plot(x[i][0], x[i][1], 'o')
# plt.show()

# for i in range(0, len(x)):
# 	if y[i] == 0:
# 		plt.plot(x[i][0], x[i][1], 'ro')
# 	else:
# 		plt.plot(x[i][0], x[i][1], 'bo')


# plt.show()



from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def getImage(path):
	return OffsetImage(plt.imread(path))

paths = ['apple.png', 'plum.png', 'emoticon.png']



x1 = [1,1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 3, 2, 1, 2, 1, 2, 5]
y1 = [2,1, 1, 3, 1, 1, 0.4, 0.5, 0.4, 2, 1, 4, 4,0,2,4,0, 2, 2]

x2 = [ 4, 5, 6, 7, 8, 10, 6]
y2 = [ 6, 7, 6, 5, 4, 3, 4]


fig, ax = plt.subplots()
ax.scatter(x1, y1)
ax.scatter(x2, y2)


for x0, y0 in zip(x1, y1):
	ab = AnnotationBbox(getImage(paths[1]), (x0, y0), frameon=False)
	ax.add_artist(ab)


for x0, y0 in zip(x2, y2):
	ab = AnnotationBbox(getImage(paths[0]), (x0, y0), frameon=False)
	ax.add_artist(ab)


# plt.show()

x3 = 7
y3 = 7

ax.scatter([x3], [y3])

ab = AnnotationBbox(getImage(paths[0]), (x3, y3), frameon=False)
ax.add_artist(ab)


plt.show()
