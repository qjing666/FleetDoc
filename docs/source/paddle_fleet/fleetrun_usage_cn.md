# fleetrun 启动分布式任务

我们提供`fleetrun`命令，只需一行简单的启动命令，即可轻松地将Paddle Fleet GPU单机单卡任务切换为多机多卡任务，也可将参数服务器单节点任务切换为多个服务节点、多个训练节点的分布式任务。

## 使用要求
使用`fleetrun`命令的要求：
- 安装 paddlepaddle 2.0-rc 及以上

## 使用说明
####  GPU场景
- **GPU单机单卡训练**

```
 fleetrun --gpus=0 train.py
 python train.py
```

注：如果指定了`export CUDA_VISIBLE_DEVICES=0` ，则可以直接使用：
```
export CUDA_VISIBLE_DEVICES=0
fleetrun train.py
```

-  **GPU单机4卡训练 **

```
fleetrun --gpus=0,1,2,3 train.py
```

注：如果指定了```export CUDA_VISIBLE_DEVICES=0,1,2,3``` ，则可以直接使用：
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
fleetrun train.py
```

-   **GPU多机多卡训练 **
 	- 2机8卡 (每个节点4卡)
```
fleetrun --ips="xx.xx.xx.xx,yy.yy.yy.yy" --gpus=0,1,2,3 train.py
```
注：如果每台机器均指定了```export CUDA_VISIBLE_DEVICES=0,1,2,3``` ，则可以直接使用：
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
fleetrun --ips="xx.xx.xx.xx,yy.yy.yy.yy" train.py
```

	- 2机16卡（每个节点8卡，假设每台机器均有8卡可使用）
```
fleetrun --ips="xx.xx.xx.xx,yy.yy.yy.yy" train.py
```

- ** GPU 在PaddleCloud上提交任务 **
**PaddleCloud**是百度开源的云上任务提交工具，提供云端训练资源，打通⽤户云端资源账号，并且支持以命令行形式进行任务提交、查看、终止等多种功能。PaddleCloud更多详情：[PaddleCloud](https://github.com/PaddlePaddle/PaddleCloud "PaddleCloud")

  在PaddleCloud上启动分布式任务十分方便，无论执行单机单卡还是多机多卡任务，只需使用：
```
fleetrun  train.py 
```

####  CPU场景

-  **参数服务器训练 - 单机训练（0个服务节点，1个训练节点） **

```
python train.py
```

-  **参数服务器训练 - 单机模拟分布式训练（1个服务节点，4个训练节点）**

```
fleetrun --server_num=1 --worker_num=4 train.py
```

-  **参数服务器训练 - 多机训练（2台节点，每台节点均有1个服务节点，4个训练节点） **

```
 # 2个servers 8个workers
 fleetrun --servers="xx.xx.xx.xx:6170,yy.yy.yy.yy:6171" --workers="xx.xx.xx.xx:6172,xx.xx.xx.xx:6173,xx.xx.xx.xx:6174,xx.xx.xx.xx:6175,yy.yy.yy.yy:6176,yy.yy.yy.yy:6177,yy.yy.yy.yy:6178,yy.yy.yy.yy:6179" train.py
```

- **参数服务器训练 - 在PaddleCloud上提交任务 **
由于Paddlecloud对参数服务器训练做了比较完备的封装，因此可以直接使用：
```
python train.py
```

## fleetrun参数介绍
- GPU模式相关参数:
	- ips （str，可选）： 指定选择哪些节点IP进行训练，默认为『127.0.0.1』, 即会在本地执行单机单卡或多卡训练。
	- gpus（str, 可选）： 指定选择哪些GPU卡进行训练，默认为None，即会选择`CUDA_VISIBLE_DEVICES`所显示的所有卡。

- 参数服务器模式可配参数:
	- server_num（int，可选）：本地模拟分布式任务中，指定参数服务器服务节点的个数
	- worker_num（int，可选）：本地模拟分布式任务中，指定参数服务器训练节点的个数
	- servers（str, 可选）： 多机分布式任务中，指定参数服务器服务节点的IP和端口
	- workers（str, 可选）： 多机分布式任务中，指定参数服务器训练节点的IP和端口

- 公用：
	- log_dir（str, 可选）： 指定分布式任务训练日志的保存路径，默认保存在"./log/"目录。


## 利用fleetrun将单机单卡任务转换为单机多卡任务
下面我们将通过例子，为您详细介绍如何利用`fleetrun`将单机单卡训练任务转换为单机多卡训练任务。
FleetX提供非常简单易用的代码来实现Imagenet数据集上训练ResNet50模型。
```py
import fleetx as X
import paddle
import paddle.distributed.fleet as fleet

configs = X.parse_train_configs()

model = X.applications.Resnet50()
imagenet_downloader = X.utils.ImageNetDownloader()
local_path = imagenet_downloader.download_from_bos(local_path='./data')
loader = model.load_imagenet_from_file(
    "{}/train.txt".format(local_path), batch_size=32)

fleet.init(is_collective=True)

optimizer = paddle.optimizer.Momentum(
    learning_rate=configs.lr,
    momentum=configs.momentum,
    regularization=paddle.fluid.regularizer.L2Decay(0.0001))
optimizer = fleet.distributed_optimizer(optimizer)
optimizer.minimize(model.loss)

trainer = X.MultiGPUTrainer()
trainer.fit(model, loader, epoch=10)
```
#### 单机单卡训练
将上述代码保存在`res_app.py`代码中，单机单卡训练十分的简单，只需要：
```
export CUDA_VISIBLE_DEVICES=0
python res_app.py
```
#### 单机多卡训练
从单机单卡训练到单机多卡训练不需要改动`res_app.py`代码，只需改一行启动命令：
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
fleetrun res_app.py
```
训练日志可以在终端上查看，也可稍后在./log/目录下查看每个卡的日志。

