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

### 3. 使用LDA算法对Iris数据集进行降维可视化

Hint: [LDA降维可视化教程](https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py)
