# RoboTwin环境配置

/home/xiongxiao/repo/vla-evaluation-harness/configs/robotwin_eval.yaml
如果你想跑多个任务（如 grab_roller + pick_apple），正确的方式是写多个 benchmark 条目：

  benchmarks:
    - benchmark: "vla_eval.benchmarks.robotwin.benchmark:RoboTwinBenchmark"
      mode: sync
      episodes_per_task: 1
      params:
        task_name: grab_roller
        task_config: demo_clean
        seed: 0
        instruction_type: seen
        test_num: 1
        skip_expert_check: true
      action_dim: 14

    - benchmark: "vla_eval.benchmarks.robotwin.benchmark:RoboTwinBenchmark"
      mode: sync
      episodes_per_task: 1
      params:
        task_name: pick_apple
        task_config: demo_clean
        seed: 0
        instruction_type: seen
        test_num: 1
        skip_expert_check: true
      action_dim: 14



## 查看docker文件内容
docker run --rm --entrypoint "" ghcr.io/allenai/vla-evaluation-harness/robotwin:latest cat /app/RoboTwin/task_config/demo_clean.yml
## 查看docker文件夹内容
docker run --rm --entrypoint "" ghcr.io/allenai/vla-evaluation-harness/robotwin:latest ls /app/RoboTwin/task_config/


## 一个task重复测试n次
修改/home/xiongxiao/repo/vla-evaluation-harness/configs/robotwin_eval.yaml 里面的test_num参数
不要修改episodes_per_task，因为这个参数就是同一个 task（同一个 seed/初始条件），用同样的设置重复跑几次，对于确定性的仿真环境（同 seed、同模型），重复跑结果一样，所以 episodes_per_task > 1 没太大意义。test_num是不同的seed


# 换task config

已经支持了。在 config YAML 的 docker 部分加 volumes 即可。比如你在宿主机有一个自定义的 task_config 文件
  /home/xiongxiao/my_task_config.yml，这样配置：

  docker:
    image: ghcr.io/allenai/vla-evaluation-harness/robotwin:latest
    volumes:
      - "/home/xiongxiao/my_task_config.yml:/app/RoboTwin/task_config/demo_clean.yml"

  这会把你宿主机的文件挂载进去，覆盖容器内的 demo_clean.yml。

  你也可以用一个新名字（不覆盖原文件），然后在 params 里改 task_config：

  docker:
    image: ghcr.io/allenai/vla-evaluation-harness/robotwin:latest
    volumes:
      - "/home/xiongxiao/my_task_config.yml:/app/RoboTwin/task_config/my_custom.yml"

  benchmarks:
    - benchmark: "vla_eval.benchmarks.robotwin.benchmark:RoboTwinBenchmark"
      mode: sync
      episodes_per_task: 1
      params:
        task_name: grab_roller
        task_config: my_custom      # 对应容器内 /app/RoboTwin/task_config/my_custom.yml
        seed: 0
        instruction_type: seen
        test_num: 1
        skip_expert_check: true
      action_dim: 14

  底层就是 docker run -v 挂载，代码在 src/vla_eval/cli/main.py:226-227。


# Episode Sharding (environment parallelism) 工作机制
1. orchestrator 把 tasks × episodes_per_task 展开成 work items（orchestrator.py:110）
2. 按 i % num_shards == shard_id 轮询分配（orchestrator.py:114）
3. 每个 shard 通过 shard_docker_flags 自动分配到一张 GPU（docker_resources.py:103-106，按 shard_id %
  len(gpu_list) 轮询）
4. 每个 shard 输出独立的 {name}_shard{id}of{total}.json 文件
5. 全部跑完后用 vla-eval merge 合并

跑法（用 GPU 4,5,6,7 跑 4 个 shard）
假设你的 config 是 configs/robotwin_eval.yaml，test_num: 100, episodes_per_task: 1，要 4 卡并行：
# 并行启动 4 个 shard，每个占一张 GPU（4→shard0, 5→shard1, 6→shard2, 7→shard3）
  for i in 0 1 2 3; do
    vla-eval run --dev \
      --config configs/robotwin_eval.yaml \
      --shard-id $i --num-shards 4 \
      --gpus 4,5,6,7 \
      > logs/shard${i}.log 2>&1 &
  done
  wait
  # 合并结果
  vla-eval merge -c configs/robotwin_eval.yaml -o results/robotwin_merged.json

  每个 shard 会处理 25 个 episode（100 / 4），分别在 GPU 4/5/6/7 上跑。

batch Batch Model Server, 需要稍微改一下model server端代码

# 终端 1：启动 batched server
  vla-eval serve --config configs/model_servers/groot_base.yaml

  # 终端 2：4 个 shard 并发
  ./scripts/run_sharded_dev.sh -c configs/robotwin_eval.yaml -n 4


我感觉skip_expert_check=false的时候，如果你开了sharding,
4 个 shard 各自会发生什么

  因为 4 个进程相互独立，每个进程的 get_tasks() 都会完整地重跑这个 while 循环：

  Shard 0: get_tasks() → 跑 100 次 oracle planner → 返回 [task0..task99] → 过滤剩 [task0,4,8...96]
   (25 个)
  Shard 1: get_tasks() → 跑 100 次 oracle planner → 返回 [task0..task99] → 过滤剩 [task1,5,9...97]
   (25 个)
  Shard 2: get_tasks() → 跑 100 次 oracle planner → 返回 [task0..task99] → 过滤剩 [task2,6,10..98]
   (25 个)
  Shard 3: get_tasks() → 跑 100 次 oracle planner → 返回 [task0..task99] → 过滤剩 [task3,7,11..99]
   (25 个)

  4 个 shard 加起来一共跑了 400 次 oracle planner，但实际只用到了 100 个任务的结果。
  第1个是效率浪费，第2个是可能task 会漏掉或者重复
  第一层：seed → success 的判定是否一致？

  get_tasks() 的循环本质上是：
  seed=100000: oracle 跑 → 成功？  →  采纳为 task[0]
  seed=100001: oracle 跑 → 失败    →  跳过
  seed=100002: oracle 跑 → 成功？  →  采纳为 task[1]
  ...

  是否一致取决于 env.plan_success and env.check_success() 在不同进程里是否给同样的结果：

  - 大概率一致：RobotWin 的 oracle planner 是 hand-scripted 策略，SAPIEN 物理引擎在固定 seed
  下是确定性的，oracle 用的是 ground-truth state（不是 rendered image），所以理论上 4 个 shard
  会得到完全相同的 100 个 seed 列表。
  - 但有风险：
    - 不同 shard 跑在不同 GPU 上（4/5/6/7），如果 oracle 内部有任何 GPU 计算（罕见，但有些
  RobotWin task 用 GPU 做 motion planning），浮点非结合性可能让边缘 case 的 success 判定翻转
    - SAPIEN 的某些版本对 CUDA non-determinism 不完全免疫


   每个 shard 独立生成自己那份 tasks 列表，然后按 index 取模。所以：

  Shard 0: 自己跑出 list_0 (100 tasks) → 取 [0, 4, 8, ..., 96] = 25 个
  Shard 1: 自己跑出 list_1 (100 tasks) → 取 [1, 5, 9, ..., 97] = 25 个
  Shard 2: 自己跑出 list_2 (100 tasks) → 取 [2, 6, 10, ..., 98] = 25 个
  Shard 3: 自己跑出 list_3 (100 tasks) → 取 [3, 7, 11, ..., 99] = 25 个

  好的情况（list_0 == list_1 == list_2 == list_3 在 seed 维度上）：4 个 shard 各拿 25 个不同 index
   的 task，加起来正好是完整的 100 个，没重没漏。但每个 task 的 instruction
  字符串可能不一样（np.random 不一致）。

  坏的情况（GPU 非确定性导致 list 在 seed 维度上也分歧）：
  - 假设 shard 0 觉得 seed 100005 失败、shard 1 觉得它成功，那么：
    - shard 0 的 list_0[5] 可能是 seed 100007（跳过了 100005）
    - shard 1 的 list_1[5] 可能是 seed 100005
  - 它们对应不同的 work_item index，可能漏掉某些 seed或重复某些 seed
  - 合并结果时你算的成功率就不对了


  No Batch Parallel Evaluation (1个仿真环境+1个batch)
  修改configs/model_servers/groot_base.yaml：
  - max_batch_size: 1
  启动model_server: vla-eval serve --config configs/model_servers/groot_base.yaml
  跑评测（不用 sharding）      
  更改/home/xiongxiao/repo/vla-evaluation-harness/configs/robotwin_eval.yaml   里面       gpus: "4"数字可为指定gpu id                                                      
                                                                                     
    vla-eval run --dev --config configs/robotwin_eval.yaml                                          
                                                                                                  
  不传 --shard-id / --num-shards，单进程顺序跑完所有 episode。结果直接写到                        
  results/RoboTwinBenchmark_sync_<timestamp>.json，不需要 merge。      