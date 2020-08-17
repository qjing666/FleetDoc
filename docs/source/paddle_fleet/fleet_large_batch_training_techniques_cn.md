# 使用超大Batch进行训练

## 简介 + strategy列表


随着训练数据规模的逐渐增加，训练更大、更深的深度学习模型成为一个主流趋势。目前的深度学习模型训练，通常要求保留前向计算的隐层结果，并且需要保存结果的数量会随着模型层数的增加线性增加，这对于目前能够使用的AI芯片的内存大小是个挑战。Fleet中提供了两种扩大训练batch大小的策略：Forward Recomputation Backpropagation (FRB) 以及 Gradient Merge。下面我们将分别对这两个策略进行讲解，并会基于BERT模型提供使用样例。
## Forward Recompute Backpropagation

### 策略简介

众所周知，深度学习网络的一轮训练迭代包含三个步骤：

- **前向计算**：运行前向算子（Operator）来计算中间隐层（Variable）的值
- **反向计算**：运行反向算子来计算参数（Parameter）的梯度
- **参数更新**：应用优化算法来更新参数的值

在前向计算过程中，前向算子会输出大量的中间计算结果，在Paddle中，会使用**Variable**来存储这些隐层的中间结果。有些中间结果在反向计算中会做为反向算子的输入，这些中间结果会被储存在内存中直到相应的反向算子计算完毕。当模型层数加深时，需要储存的中间结果数量可达成千上万个，占据大量的内存。

Forward Recomputation Backpropagation（FRB）的思想是将深度学习网络切分为k个部分（segments）。对每个segment而言：前向计算时，除了小部分必须存储在内存中的Variable外(我们后续会讨论这些特殊Variable)，其他中间结果都将被删除；在反向计算中，首先重新计算一遍前向算子，以获得中间结果，再运行反向算子。简而言之，FRB和普通的网络迭代相比，多计算了一遍前向算子。

我们把切分网络的变量叫做checkpoints。那么该如何选择这些checkpoints呢？我们知道深度学习网络通常是由一个个模块串联得到的，比如ResNet-50由16个block串联而成， Bert-Large由24个transformer串联而成，以两个子模块中间的变量作为切分点就是一个很好的选择。 对于非串联的网络（比如含有大量shortcut结构的网络），FRB也支持对其做切分， 只是可能多耗费一点内存（用于存储shortcut的Variable）。同时我们也可以通过一些动态规划的算法，根据指定的内存自动搜索合适的checkpoints，来支持各种网络结构。

下图是由4个fc Layer、3个relu Layer、1个sigmoid Layer和1个log-loss Layer串联而成的一个网络：最左侧为其前向计算流程、中间是普通的前向计算和反向计算流程、最右侧为添加FRB后的前向计算和反向计算流程。其中方框代表算子(Operator)，红点代表前向计算的中间结果、蓝点代表checkpoints。

<img src='./img/recompute.png' width = "1000" height = "584" align="middle"/>

添加FRB后，前向计算中需要存储的中间Variable从4个(红点)变为2个(蓝点)， 从而节省了这部分内存。当然了，重计算的部分也产生了新的中间变量， 这就需要根据实际情况来做权衡了。这个例子里的网络比较浅，通常来讲， 对层数较深的网络，FRB节省的内存要远多于新增加的内存。

通过在BERT模型上的测试，Recompute可将batch size扩大近三倍。同时也可以配合混合精度使用来进一步提升batch size及训练速度。

- **Bert_large**: 

|Model|Baseline|Recompute| Recompute + mixed precision|
|:---:|:---:|:---:|:---:|
|batch size| 14 | 53 | 87 |
|speed|18.2 sents/s| 12.88 sents/s| 19.14 sents/s |


### 应用实例

首先，下载训练所用到的数据及词表
```sh
wget --no-check-certificate https://fleet.bj.bcebos.com/Bertdata/train_data.tar.gz
tar -xf train_data.tar.gz
wget --no-check-certificate https://fleet.bj.bcebos.com/Bertdata/vocab.txt
```

然后我们就可以使用fleet API完成BERT的分布式训练程序了（假设脚本名称为bert_app.py）：
#### 添加依赖

```python
import os
import fleetx as X
import paddle.fluid as fluid
import paddle.distributed.fleet.base.role_maker as role_maker
import time
import paddle.distributed.fleet as fleet
```

#### 初始化
```python
configs = X.parse_train_configs()
role = role_maker.PaddleCloudRoleMaker(is_collective=True)
fleet.init(role)
```
#### 加载模型及数据
```
model = X.applications.Bert_large()

data_loader = model.load_digital_dataset_from_file(
    data_dir='./train_data',
    vocab_path='./vocab.txt',
    max_seq_len=512,
    batch_size=53,
)
```

#### 定义strategy以及optimizer

```python
dist_strategy = fleet.DistributedStrategy()
# 使用Recompute，并设置checkpoints
dist_strategy.recompute = True
dist_strategy.recompute_configs = {"checkpoints": model.checkpoints}

optimizer = fluid.optimizer.Adam(learning_rate=configs.lr)
optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
optimizer.minimize(model.loss)
```

#### 开始训练
```python
place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

total_time = 0
for i, data in enumerate(data_loader()):
    if i >= 10:
        start_time = time.time()
    cost_val = exe.run(fluid.default_main_program(),
                       feed=data,
                       fetch_list=[model.loss.name])
    if i >= 10:
        end_time = time.time()
        total_time += (end_time - start_time)
        print(
            "worker_index: %d, step%d cost = %f, total time cost = %f, step per second: %f, speed: %f"
            % (fleet.worker_index(), i, cost_val[0], total_time,
               (i - 9) / total_time, 1 / (end_time - start_time)))
```
完成脚本的编写后我们就可以使用以下命令开始训练：

```sh
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


