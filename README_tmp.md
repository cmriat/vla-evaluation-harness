

batch Batch Model Server, 需要稍微改一下model server端代码

# 终端 1：启动 batched server
  vla-eval serve --config configs/model_servers/groot_base.yaml

  # 终端 2：4 个 shard 并发
  ./scripts/run_sharded_dev.sh -c configs/robotwin_eval.yaml -n 4





  
