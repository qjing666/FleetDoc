# 使用超大Batch进行训练

## 简介 + strategy列表


随着训练数据规模的逐渐增加，训练更大、更深的深度学习模型成为一个主流趋势。目前的深度学习模型训练，通常要求保留前向计算的隐层结果，并且需要保存结果的数量会随着模型层数的增加线性增加，这对于目前能够使用的AI芯片的内存大小是个挑战。Fleet中提供了两种扩大训练batch大小的策略：Forward Recomputation Backpropagation (FRB) 以及 Gradient Merge。下面我们将分别对这两个策略进行讲解，并会基于BERT模型提供使用样例。

在开始之前，我们需要准备训练数据及词表

```sh
wget --no-check-certificate https://fleet.bj.bcebos.com/Bertdata/train_data.tar.gz
tar -xf train_data.tar.gz
wget --no-check-certificate https://fleet.bj.bcebos.com/Bertdata/vocab.txt
```
## Forward Recompute Backpropagation

首先，我们来介绍Fleet中通过 Forward Recompute Backpropagation 策略增大 BERT 模型在分布式训练中 batch size 的方法（假设脚本名称为bert_app.py）：

### 添加依赖

```python
import os
import time
import paddle
import fleetx as X
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
```

### 定义分布式模式并初始化

通过`X.parse_train_configs()`接口，用户可以定义训练相关的参数，如：学习率、衰减率等。同时通过`fleet.init()`接口定义了分布式模型，下面代码中的`is_collective=True`表示采用集合通信的GPU分布式模式训练模型。
```python
configs = X.parse_train_configs()
role = role_maker.PaddleCloudRoleMaker(is_collective=True)
fleet.init(role)
```

### 加载模型及数据

用户可以通过`X.applications`接口加载我们预先定义好的模型，如：Resnet50、VGG16、BERT等。并使用定制化的data_loader加载模型，同时可以定义训练中使用的batch_size等参数。下面的例子中，我们使用了recompute对Bert_large模型所支持的最大batch_size -- 53
```python
model = X.applications.Bert_large()

data_loader = model.load_digital_dataset_from_file(
    data_dir='./train_data',
    vocab_path='./vocab.txt',
    max_seq_len=512,
    batch_size=53,
)
```

### 定义recompute strategy 以及 optimizer

接下来我们就可以定义分布式训练中所应用到的策略了。Forward Recomputation Backpropagation（FRB）的思想是将深度学习网络切分为k个部分（segments）。对每个segment而言：前向计算时，除了小部分必须存储在内存中的Variable外，其他中间结果都将被删除；在反向计算中，首先重新计算一遍前向算子，以获得中间结果，再运行反向算子。简而言之，FRB和普通的网络迭代相比，多计算了一遍前向算子。

我们把切分网络的变量叫做checkpoints。那么该如何选择这些checkpoints呢？我们知道深度学习网络通常是由一个个模块串联得到的，比如ResNet-50由16个block串联而成， Bert-Large由24个transformer串联而成，以两个子模块中间的变量作为切分点就是一个很好的选择。 对于非串联的网络（比如含有大量shortcut结构的网络），FRB也支持对其做切分， 只是可能多耗费一点内存（用于存储shortcut的Variable）。同时我们也可以通过一些动态规划的算法，根据指定的内存自动搜索合适的checkpoints，来支持各种网络结构。

下面的例子中，为了使用Recompute策略，我们将`dist_strategy.recompute`设置为True 并设置我们事先定义好的checkpoints。

接下来用户需要定义训练中更新模型所用到的优化器，并使用`fleet.distributed_optimizer`接口将优化器转换为分布式模式。

最后运行`optimizer.minimize(model.loss)` 将反向计算的算子插入训练网络，我们就可以开始训练了。
```python
dist_strategy = fleet.DistributedStrategy()
# 使用Recompute，并设置checkpoints
dist_strategy.recompute = True
dist_strategy.recompute_configs = {"checkpoints": model.checkpoints}

optimizer = fluid.optimizer.Adam(learning_rate=configs.lr)
optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
optimizer.minimize(model.loss)
```

### 开始训练

```python
place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

total_time = 0
for i, data in enumerate(data_loader()):
    start_time = time.time()
    cost_val = exe.run(fluid.default_main_program(),
                       feed=data,
                       fetch_list=[model.loss.name])
    end_time = time.time()
    total_time += (end_time - start_time)
    print(
        "worker_index: %d, step%d cost = %f, total time cost = %f, step per second: %f, speed: %f"
        % (fleet.worker_index(), i, cost_val[0], total_time,
           (i - 9) / total_time, 1 / (end_time - start_time)))
```

### 运行训练脚本
完成脚本的编写后我们就可以使用以下命令训练分布式模型：
```sh
fleetrun --gpus 0,1,2,3,4,5,6,7 bert_recompute.py
```
### 策略简介

- **Bert_large**: 

|Model|Baseline|Recompute| Recompute + mixed precision|
|:---:|:---:|:---:|:---:|
|batch size| 14 | 53 | 87 |
|speed|18.2 sents/s| 12.88 sents/s| 19.14 sents/s |



fleetrun --gpus 0,1,2,3,4,5,6,7 bert_recompute.py
```

## Gradient Merge

在分布式训练中，经常遇到显存或者内存不足的情况，该问题通常由以下原因导致：

- 输入过大（batch size过大或视频等较大的数据）
- 中间层输出占据的显存超出了显存大小
- 参数过大（如CPU的embedding）

GradientMerge 策略的做法为：将大batch的输入切分成若干小batch，并对这些小batch分别进行 "前向+反向" 网络计算从而得到梯度。其间会有一部分显存/内存用于存放梯度，对每个小batch计算出的梯度进行叠加，在计算完所有小batch后用累加的梯度对模型进行更新。
通过GradientMerge 策略，用户只需要定义大batch被分割的粒度便可以实现大batch训练的目的。

### 应用样例


