### 1. 实现可以用于[Breast Cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer)的线性分类器，使用 auc, acc, precision, recall, f1 作为评价指标。

Hint: [分类器](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)

```python
from sklearn import linear_model
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

data = load_breast_cancer()
clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(data.data[:451], data.target[:451])
print(roc_auc_score(data.target[451:], clf.predict(data.data[451:])))
print(accuracy_score(data.target[451:], clf.predict(data.data[451:])))
print(precision_recall_fscore_support(data.target[451:], clf.predict(data.data[451:]), average='binary'))
```

### 2. 尝试实现一个基于伪逆方法的线性回归器，并和[示例代码](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)中的模型做比较。

Hint: [numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html)

notes

伪拟（[The Moore Penrose Pseudoinverse](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.9-The-Moore-Penrose-Pseudoinverse/)）

(没明白题目中说的“比较”，就只用伪逆造回归器，不比较了🙃)

```python
import numpy as np

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
A = np.hstack((X, np.ones((np.shape(X)[0], 1))))
A_plus = np.linalg.pinv(A)
coefs = A_plus.dot(y)
print(coefs)
```

### 3. 使用LDA算法对Iris数据集进行降维可视化

Hint: [LDA降维可视化教程](https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py)

notes

[Numpy数组的布尔索引和花式索引](https://www.jianshu.com/p/743b3bb340f6)

```python
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)
colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')
plt.show()
```
