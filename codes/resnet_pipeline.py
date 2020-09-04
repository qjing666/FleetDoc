import math 
import fleetx as X
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
from fleetx.dataset.image_dataset import image_dataloader_from_filelist

def conv_bn_layer(input,
                  num_filters,
                  filter_size,
                  stride=1,
                  groups=1,
                  act=None):
    conv = fluid.layers.conv2d(input=input,
                               num_filters=num_filters,
                               filter_size=filter_size,
                               stride=stride,
                               padding=(filter_size-1)//2,
                               groups=groups,
                               act=None,
                               bias_attr=False)
    return fluid.layers.batch_norm(input=conv,
                                   act=act)
 
def shortcut(input,
             ch_out,
             stride,
             is_first):
    ch_in = input.shape[1]
    if ch_in != ch_out or stride != 1 or is_first == True:
        return conv_bn_layer(input,
                             ch_out,
                             1,
                             stride)
    else:
        return input
 
 
def bottleneck_block(input,
                     num_filters,
                     stride):
    conv0 = conv_bn_layer(input=input,
                          num_filters=num_filters,
                          filter_size=1,
                          act='relu')
    conv1 = conv_bn_layer(input=conv0,
                          num_filters=num_filters,
                          filter_size=3,
                          stride=stride,
                          act='relu')
    conv2 = conv_bn_layer(input=conv1,
                          num_filters=num_filters*4,
                          filter_size=1,
                          act=None)
 
    short = shortcut(input,
                     num_filters*4,
                     stride,
                     is_first=False)
 
    return fluid.layers.elementwise_add(x=short,
                                        y=conv2,
                                        act='relu')



def build_network(input,
                  layers=50,
                  class_dim=1000):
    supported_layers = [18, 34, 50, 101, 152]
    assert layers in supported_layers
    depth = None
    if layers == 18:
        depth = [2, 2, 2, 2]
    elif layers == 34 or layers == 50:
        depth = [3, 4, 6, 3]
    elif layers == 101:
        depth = [3, 4, 23, 3]
    elif layers == 152:
        depth = [3, 8, 36, 3]
    num_filters = [64, 128, 256, 512]
    offset = 0
    with fluid.device_guard("gpu:%d"%(offset)):
        conv = conv_bn_layer(input=input,
                             num_filters=64,
                             filter_size=7,
                             stride=2,
                             act='relu')
        conv = fluid.layers.pool2d(input=conv,
                                   pool_size=3,
                                   pool_stride=2,
                                   pool_padding=1,
                                   pool_type='max')
    offset += 1
    if layers >= 50:
        for block in range(len(depth)):
            with fluid.device_guard("gpu:%d"%(offset)):
                for i in range(depth[block]):
                    conv = bottleneck_block(
                            input=conv,
                            num_filters=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1)
            offset += 1

        with fluid.device_guard("gpu:%d"%(offset)):
            pool = fluid.layers.pool2d(input=conv,
                                       pool_size=7,
                                       pool_type='avg',
                                       global_pooling=True)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = fluid.layers.fc(
                    input=pool,
                    size=class_dim,
                    param_attr=fluid.param_attr.ParamAttr(
                        initializer=fluid.initializer.Uniform(-stdv, stdv)))
    else:
        for block in range(len(depth)):
            with fluid.device_guard("gpu:%d"%(offset)):
                for i in range(depth[block]):
                    conv = basic_block(input=conv,
                                       num_filters=num_filters[block],
                                       stride=2 if i == 0 and block != 0 else 1,
                                       is_first=block==i==0)
            offset += 1
        with fluid.device_guard("gpu:%d"%(offset)):
            pool = fluid.layers.pool2d(input=conv,
                                       pool_size=7,
                                       pool_type='avg',
                                       global_pooling=True)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = fluid.layers.fc(
                    input=pool,
                    size=class_dim,
                    param_attr=fluid.param_attr.ParamAttr(
                        initializer=fluid.initializer.Uniform(-stdv, stdv)))
    return out, offset

with fluid.device_guard("gpu:0"):
    image_shape = [3, 224, 224]
    image = fluid.layers.data(name="feed_image",
                              shape=image_shape,
                              dtype="float32")
    label = fluid.layers.data(name="feed_label", shape=[1], dtype="int64")
    data_loader = fluid.io.DataLoader.from_generator(
            feed_list=[image, label],
            capacity=64,
            use_double_buffer=True,
            iterable=False)

fc, offset = build_network(image)

with fluid.device_guard("gpu:%d"%(offset)):
    out, prob = fluid.layers.softmax_with_cross_entropy(logits=fc,
                                                        label=label,
                                                        return_softmax=True)
    loss = fluid.layers.mean(out)
    acc_top1 = fluid.layers.accuracy(input=prob, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=prob, label=label, k=5)

#fleet.init(is_collective=True) 
strategy = fleet.DistributedStrategy()
strategy.pipeline = True
strategy.pipeline_configs = {"micro_batch": 12}
optimizer = fluid.optimizer.Momentum(0.1, momentum=0.9)
optimizer = fleet.distributed_optimizer(optimizer, strategy)
optimizer.minimize(loss)

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

loader = image_dataloader_from_filelist(
    "/ssd2/lilong/ImageNet/train.txt", inputs=['feed_image', 'feed_label'], batch_size=48)

for data in loader:
    metrices = exe.run(fluid.default_main_program(), feed=data, fetch_list=['mean_0.tmp_0'])
