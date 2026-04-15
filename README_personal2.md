# 测试 RoboTwin 仿真环境流程

## 1. 总体测试流程

### 1.1 修改 model server 配置(coder workspace)

编辑 `configs/model_servers/groot_base.yaml`，修改以下参数：
- `port`: 端口号（多实例时每个用不同端口）
- 其他参数按需调整

### 1.2 启动 server(coder workspace)
> `CUDA_VISIBLE_DEVICES` 指定模型加载到哪块/哪些 GPU。
```zsh
CUDA_VISIBLE_DEVICES=0 vla-eval serve -c configs/model_servers/groot_base.yaml
```

### 1.3 获取 Model Server 地址
1. 打开 https://coder.lionrock.com/，找到目标 Workspace
2. 点击 **Open Ports**
3. 找到 model server 对应的端口
4. 点击 **Share this port**，将 Sharing Level 改为 **Public**
5. 右键复制链接地址：比如8000--main--xiaoxiong-8gpu-0--xiaoxiong-sherry.coder.lionrock.com"

### 1.4 配置 hosts 解析(4090裸金属root权限)
编辑 `/etc/hosts`，添加：
```
106.13.249.94 8000--main--xiaoxiong-8gpu-0--xiaoxiong-sherry.coder.lionrock.com
```

### 1.5 修改 Benchmark Client 配置(4090裸金属user权限)
编辑 `configs/robotwin_eval.yaml`，修改以下参数：
- server.url: 指向 model server
  - Example: "wss://8000--main--xiaoxiong-8gpu-0--xiaoxiong-sherry.coder.lionrock.com"
- test_num: 评测任务的 episode 数量，每个 episode 使用不同的 seed（`100000*(1+seed) + i`）
- episodes_per_task: 同一个 task（同一个 seed/episode初始条件）重复跑几次。对于确定性仿真环境（相同 seed + 相同模型），重复跑结果一样，所以一般保持 `1` 即可。
- task_config: 如果宿主机上有自定义的 task_config 文件（假设文件路径为/home/xiongxiao/my_task_config.yml），通过 Docker volume 挂载进容器：
  - 修改docker.volumes参数："/home/xiongxiao/my_task_config.yml:/app/RoboTwin/task_config/my_task_config.yml"
  - 修改task_config参数: my_task_config
- 其他参数按需调整

### 1.6 启动 Benchmark Client(4090裸金属user权限)

## 2. 三种测试方式

> **前提**：无论使用哪种方法，都需要包含上面第 1 节的所有步骤。

### 2.1 方法一: 单进程顺序评测（1 个仿真环境 + batch_size=1）

> 最简单的方式，单进程依次跑完所有 episode。

#### 修改 Model server 配置(coder workspace)
编辑`configs/model_servers/groot_base.yaml`，修改以下参数：
- max_batch_size: 1
- max_wait_time: 随便设置(不会用到)

#### 启动 Model server(coder workspace)
```zsh
CUDA_VISIBLE_DEVICES=0 vla-eval serve -c configs/model_servers/groot_base.yaml
```

#### 修改 Benchmark Client 配置(4090裸金属user权限)
编辑 `configs/robotwin_eval.yaml`，修改以下参数：
- docker.gpus: "5" -> "指定gpu ID"

#### 启动 Benchmark Client(4090裸金属user权限)
```zsh
vla-eval run --dev --config configs/robotwin_eval.yaml
```


### 2.2 方法二: Episode Sharding（多仿真环境并行 + batch_size=1）
> 将 episodes 分片到多个 shard（仿真环境）并行跑，所有容器连接同一个 model server。不让model server在等待仿真环境反馈的时候空闲下来。
#### 2.2.1 Episode Sharding工作机制
- orchestrator 把 test_num × episodes_per_task 展开成 work items（orchestrator.py:110）
- 按 i % num_shards == shard_id 轮询分配（orchestrator.py:114）
- 每个 shard 通过 shard_docker_flags 自动分配到一张 GPU（docker_resources.py:103-106，按 shard_id % len(gpu_list) 轮询）
- 每个 shard 输出独立的 {name}_shard{id}of{total}.json 文件
- 全部跑完后用 vla-eval merge 合并

#### 2.2.2 启动Episode Sharding测试流程
##### 2.2.2.1 启动 Model server(coder workspace)
```zsh
CUDA_VISIBLE_DEVICES=0 vla-eval serve -c configs/model_servers/groot_base.yaml
```
##### 2.2.2.2 修改 Benchmark Client 配置(4090裸金属user权限)
编辑 `configs/robotwin_eval.yaml`，修改以下参数：
- docker.gpus: "4,5,6,7" -> 用4张卡来启动多个进程（一张卡可以有多个进程）

##### 2.2.2.3 启动 Benchmark Client(4090裸金属user权限)
```zsh
./scripts/run_sharded_dev.sh -c configs/robotwin_eval.yaml -n 4  # 分4个shard
```


### 2.3 方法三: Episode Sharding + Batched Inference（多仿真环境并行 + 多 batch_size）
> 在方法 2 的基础上, `model server` 开启 `batching`，将多个仿真环境的推理请求凑成一个 batch 一起推理。适合希望同时最大化仿真并行度和 GPU 利用率的场景。

#### 2.3.1 修改 Model Server 代码
编辑`src/vla_eval/model_servers/groot_base.py`，需要有`predict_batch`函数。

#### 2.3.2 修改 Model server 配置(coder workspace)
编辑`configs/model_servers/groot_base.yaml`，修改以下参数：
- max_batch_size: 4（小于等于仿真环境shard数量）
- max_wait_time: 0.05

#### 2.3.3 启动 Model server(coder workspace)
```zsh
CUDA_VISIBLE_DEVICES=0 vla-eval serve -c configs/model_servers/groot_base.yaml
```
#### 2.3.4 修改 Benchmark Client 配置(4090裸金属user权限)
编辑 `configs/robotwin_eval.yaml`，修改以下参数：
- docker.gpus: "4,5,6,7" -> 用4张卡来启动多个进程（一张卡可以有多个进程）

#### 2.3.5 启动 Benchmark Client(4090裸金属user权限)
```zsh
./scripts/run_sharded_dev.sh -c configs/robotwin_eval.yaml -n 4  # 分4个shard
```

#### 2.3.6 方法 2 vs 方法 3 的区别
- 方法 2 中多个`仿真环境`的推理请求是串行到达`server`的（先到先推理）；
- 方法 3 中`server`会等一小段时间（`max_wait_time`）把多个请求凑成一个`batch`，一次 GPU forward pass 处理，吞吐量更高。


