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

Forward Recomputation Backpropagation（FRB）的思想是将深度学习网络切分为k个部分（segments）。


### Recompute 效果

### 效果复现的例子
添加依赖

```python
import numpy as np
import fleet_lightning as lighting
import paddle.fluid as fluid
import paddle.fleet.base.role_maker as role_maker
import time
import paddle.fleet as fleet
import paddle
```
初始化
```python
configs = lighting.parse_train_configs()
role = role_maker.PaddleCloudRoleMaker(is_collective=True)
fleet.init(role)
```
加载网络（lightning）
```
model = lighting.applications.Bert_large()
#model = lighting.applications.Bert_base()

data_loader = model.load_digital_dataset_from_file(
    data_dir='/home/mapingshuo/Fleet/benchmark/collective/bert/data/train/', 
    vocab_path='/home/mapingshuo/Fleet/benchmark/collective/bert/uncased_L-24_H-1024_A-16//vocab.txt',
    max_seq_len=512,
    batch_size=12,
)
```
定义strategy以及optimizer

```python
place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
exec_strategy = fluid.ExecutionStrategy()
exec_strategy.num_threads = 2
exec_strategy.num_iteration_per_drop_scope = 1
dist_strategy = fleet.DistributedStrategy()
dist_strategy.execution_strategy = exec_strategy
dist_strategy.recompute = False
dist_strategy.nccl_comm_num = 3
dist_strategy.use_hierarchical_allreduce = True

optimizer = fluid.optimizer.Adam(learning_rate=configs.lr)
optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
optimizer.minimize(model.loss)
```
开始训练
```python
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

for i, data in enumerate(data_loader()):
    if i >= 10:
        start_time = time.time()
    cost_val = exe.run(paddle.default_main_program(),
                       feed=data,
                       fetch_list=[model.loss.name])
    if i >= 10:
        end_time = time.time()
        total_time += (end_time - start_time)
        print(
            "worker_index: %d, step%d cost = %f, total time cost = %f, step per second: %f, speed: %f"
            % (fleet.worker_index(), i, cost_val[0], total_time,
               (i - 9) / total_time, 1 / (end_time - start_time)))
    print("step: %d, encoder_layer_16_ffn_fc_1.w_0: %s" % (
         i, scope.var("encoder_layer_16_ffn_fc_1.w_0").get_tensor().__array__()))
```


## Gradient Merge

