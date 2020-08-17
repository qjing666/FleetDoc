import os
import fleetx as X
import paddle.fluid as fluid
import paddle.distributed.fleet.base.role_maker as role_maker
import time
import paddle.distributed.fleet as fleet


configs = X.parse_train_configs()
role = role_maker.PaddleCloudRoleMaker(is_collective=True)
fleet.init(role)
model = X.applications.Bert_large()

data_loader = model.load_digital_dataset_from_file(
    data_dir='./train_data',
    vocab_path='./vocab.txt',
    max_seq_len=512,
    batch_size=13,
)

dist_strategy = fleet.DistributedStrategy()
dist_strategy.gradient_merge = True
dist_strategy.gradient_merge_configs = {"k_steps": 4, "avg": True}
optimizer = fluid.optimizer.Adam(learning_rate=configs.lr)
optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
optimizer.minimize(model.loss)

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
