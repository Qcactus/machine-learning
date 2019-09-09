### 1. 验证 AUC=0.5 的含义
```python
from sklearn.metrics import roc_curve, auc
import random
import matplotlib.pyplot as plt

random.seed(10)
sample_num = 1000
y_true = [random.randint(0, 1) for _ in range(sample_num)]
y_pred = [random.random() for _ in range(sample_num)]
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
plt.plot(fpr, tpr, label='AUC = %0.2f' % auc (fpr, tpr))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
```
### 2. 计算下面给出的二分类结果数据的 Accuracy(ACC), Precision(P), Recall(R) 和 F1(F)
```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random

random.seed(10)  # 设定随机数种子，使得每次随机结果相同，方便重复检验
sample_num = 1000
threshold = 0.6
y_true = [random.randint(0, 1) for _ in range(sample_num)]
y_pred = [random.random() for _ in range(sample_num)]
for i in range(len(y_pred)):
    if y_pred[i] > threshold:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
print(accuracy_score(y_true, y_pred))
print(precision_recall_fscore_support(y_true, y_pred, average='binary'))
```
### 3. 计算下面给出的三分类结果数据的 Accuracy(ACC), Precision(P), Recall(R) 和 F1(F)
```python
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

random.seed(10)
class_num = 3
sample_num = 1000
y_true = [random.randint(1, class_num) for _ in range(sample_num)]
y_pred = [[random.random() for _ in range(class_num)] for _ in range(sample_num)]
y_cate = []
for x in y_pred:
    y_cate.append(x.index(max(x)) + 1)
print(accuracy_score(y_true, y_cate))
print(precision_recall_fscore_support(y_true, y_cate, average='macro'))
print(precision_recall_fscore_support(y_true, y_cate, average='micro'))
print(precision_recall_fscore_support(y_true, y_cate, average='weighted'))
```
### 4. 将Iris数据集使用 k-flod 方法进行划分
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold

data = load_iris()
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(data.data):
    print("TRAIN:", train_index, "TEST:", test_index, '\n')
```
