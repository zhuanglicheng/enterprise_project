from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier

# 获取数据
mnist=fetch_mldata("MNIST original",data_home='D:\\ProjectAll\\PythonGitProjectAll\\enterprise_project')
# print(mnist)
# 查看数据集
X,y = mnist["data"],mnist["target"]
print(X.shape)
print(y.shape)
print(y[36000])

# some_digit = X[36000]
# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
#            interpolation="nearest")
# plt.axis("off")
# plt.show()

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# 训练一个二元分类器
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
y_pre = sgd_clf.predict([X_train[5]])
print(y_pre)
print(y_train[5])
