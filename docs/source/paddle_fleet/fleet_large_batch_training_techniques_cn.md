# 使用超大Batch进行训练

## 简介 + strategy列表

## Recompute

### Recompute 介绍

### Recompute 效果

### 效果复现的例子
添加依赖

```
import numpy as np
import fleet_lightning as lighting
import paddle.fluid as fluid
import paddle.fleet.base.role_maker as role_maker
import time
import paddle.fleet as fleet
import paddle
```
初始化
```
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

```
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
```
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


## Lars

