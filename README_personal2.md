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