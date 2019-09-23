### 1. 试搭建一个简单的神经网络(比如lenet)对MNIST数据集进行分类任务（Hint：tensorflow, keras, pytorch, mxnet）。

notes

主要👉[Handwritten Digit Recognition](./docs/mnist.ipynb)

```python
import mxnet as mx
def data_xform(data):
    """Move channel axis to the beginning, cast to float32, and normalize to [0, 1]."""
    return nd.moveaxis(data, 2, 0).astype('float32') / 255

train_data = mx.gluon.data.vision.MNIST(train=True).transform_first(data_xform)
val_data = mx.gluon.data.vision.MNIST(train=False).transform_first(data_xform)
batch_size = 100
train_loader = mx.gluon.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = mx.gluon.data.DataLoader(val_data, shuffle=False, batch_size=batch_size)
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
lenet = nn.HybridSequential(prefix='LeNet_')
with lenet.name_scope():
    lenet.add(
        nn.Conv2D(channels=20, kernel_size=(5, 5), activation='tanh'),
        nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        nn.Conv2D(channels=50, kernel_size=(5, 5), activation='tanh'),
        nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        nn.Flatten(),
        nn.Dense(500, activation='tanh'),
        nn.Dense(10, activation=None),
    )

lenet.initialize(mx.init.Xavier())

trainer = gluon.Trainer(
    params=lenet.collect_params(),
    optimizer='sgd',
    optimizer_params={'learning_rate': 0.04},
)
metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
num_epochs = 10

for epoch in range(num_epochs):
    for inputs, labels in train_loader:

        with autograd.record():
            outputs = lenet(inputs)
            loss = loss_function(outputs, labels)

        loss.backward()
        metric.update(labels, outputs)

        trainer.step(batch_size=inputs.shape[0])

    name, acc = metric.get()
    print('After epoch {}: {} = {}'.format(epoch + 1, name, acc))
    metric.reset()

for inputs, labels in val_loader:
    metric.update(labels, lenet(inputs))
print('Validaton: {} = {}'.format(*metric.get()))
assert metric.get()[1] > 0.985
```
### 2. 在1的基础上，通过加入dropout，使用不同的优化器，加入正则化等方式进行实验并记录，观察效果。
