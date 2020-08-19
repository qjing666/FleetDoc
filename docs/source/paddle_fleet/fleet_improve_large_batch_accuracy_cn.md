# 使用LARS / LAMB 加速分布式超大batch 训练

## 简介 
在数据并行分布式训练场景中, 常使用增加GPU数量的方式来加速训练. 为了保证GPU的算力得到充分利用, 每张GPU卡上的batch size都需要足够大. 因此当GPU 数量增加时, 训练的全局batch size 也会变大. 但越大的全batch size 训练的收敛问题[[1]](https://arxiv.org/abs/1404.5997)  [[2]](https://arxiv.org/abs/1609.04836):
 * 模型最终精度损失
 * 收敛速度变慢, 需要更多的epoch 才能收敛 

LARS[[3]](https://arxiv.org/abs/1708.03888) 和 LAMB[[4]](https://arxiv.org/abs/1904.00962) 两个优化策略常用来解决上述超大batch 训练中的收敛问题. FleetX 实现了这两种优化策略, 并提供简单易用API 接口. 通过这两个优化策略, 我们在超大batch 场景中实现了更快的收敛速度和无损的精度, 结合FleetX 中其他的策略(AMP (LINK to amp)) 极大缩短的训练整体的time2train. 中下文将通过一个简单例子介绍如何在Fleet 数据并行训练框架中使用 LARS 和LAMB, 另外给出我们使用 FleetX 实践的效果和代码.

## Fleet 效果
|     |Global batch size|epoch| top1 |
|:---:|:---:|:---:|:---:|
|[Goyal et al](https://arxiv.org/abs/1706.02677)| 8k | 90 | 76.3% |
|[LARS](https://arxiv.org/abs/1708.03888)| 32k | 90 | 72.3% |
|[FleetX lars + amp](https://LINK_to_example_code) |16k | 60 | 75.9%|
|[FleetX lars + amp](https://LINK_to_example_code) |32k | TBA | TBA |
|[FleetX lars + amp](https://LINK_to_example_code) |64k | TBA | TBA |


## LARS 
我们以在单机多卡上Resent50 训练为简单例子介绍FleetX 中 lars的用法.

#### 构建模型
首先我们要导入依赖和定义模型和 data loader, 这一步和FleetX 下其他任务基本一致.

.. code-block:: python
import os
import fleetx as X
import paddle.fluid as fluid
import paddle.distributed.fleet.base.role_maker as role_maker
import time
import paddle.distributed.fleet as fleet

model = X.applications.Resnet50()
loader = model.load_imagenet_from_file("/pathto/ImageNet/train.txt")
..

#### 定义分布式 和LARS 相关策略
LARS 优化算法的公式如下:

.. math::
    & local\_learning\_rate = learning\_rate * lars\_coeff * \
    \frac{||param||}{||gradient|| + lars\_weight\_decay * ||param||}\\
    & velocity = mu * velocity + local\_learning\_rate * (gradient + lars\_weight\_decay * param)\\
    & param = param - velocity
..

可以看到LARS 其实是在 带`weight decay` 的`momentum` 优化器的基础上加入了`local learning rate` 的逻辑, 对每一层的`learning rate` 进行了放缩. FleetX 将 LARS实现为一个 fleet meta optimizer, 在使用时需要注意一下几点:
1. LARS meta optimizer 的 inner optimizer 必须为 `momentum`, 并在 momentum 中定义 `mu` 和`lr` 参数.
2. 在 fleet dist_strategy 定义LARS 特有的 `lars_coeff` 参数和 `lars_weight_decay` 参数.
3. Weight Decay
    * LARS 已经将 `weight decay` 包含进公式中, 用户不需要再另外设置 `weight decay`.
    * FleetX 中还提供 lars_weight_decay 过滤策略, 可以通过在`exclude_from_weight_decay` 参数加入对应layer 的 `name string`, 让这一 layer 的参数不进行 lars weight decay. (通常我们将`BN` 参数 和 `FC_bias` 从lars weight decay 中过滤)

.. code-block:: python
    # FleetX 
    configs = X.parse_train_configs()
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)
    dist_strategy = fleet.DistributedStrategy()

    # 
    dist_strategy.lars = True
    dist_strategy.lars_configs = {
                        "lars_coeff": 0.001,
                        "lars_weight_decay": 0.0005,
                        "exclude_from_weight_decay": ['batch_norm', '.b_0']
                    }

    optimizer = paddle.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
    optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
    optimizer.minimize(model.loss)
..

#### 开始训练
这一部分和FleetX 中其他任务基本相同:

.. code-block:: python
place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

total_time = 0
for i, data in enumerate(data_loader()):
    start_time = time.time()
    cost_val = exe.run(paddle.static.default_main_program(),
                       feed=data,
                       fetch_list=[model.loss.name])
    end_time = time.time()
    total_time += (end_time - start_time)
    print(
        "worker_index: %d, step%d cost = %f, total time cost = %f, step per second: %f, speed: %f"
        % (fleet.worker_index(), i, cost_val[0], total_time,
           (i - 9) / total_time, 1 / (end_time - start_time))
..
