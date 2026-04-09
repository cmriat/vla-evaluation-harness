# 在4090裸金属上跑 vla-evaluation-harness 的环境准备笔记

## 1. 安装`uv`
先安装`uv`：
```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```
安装完成后检查版本：
```zsh
uv --version
```

## 2. 获取并安装`vla-evaluation-harness`
先拉取仓库并进入项目目录：
```zsh
git clone https://github.com/allenai/vla-evaluation-harness.git
cd vla-evaluation-harness
```
然后用 Python 3.11 创建环境并安装依赖：
```zsh
uv sync --python 3.11 --all-extras --dev
```
激活虚拟环境：
```zsh
source .venv/bin/activate
```
验证命令是否存在：
```zsh
vla-eval --help
```

## 3. 安装 Docker
> 这一部分通常需要在`root`权限下执行。

### 3.1 更新系统依赖
```zsh
sudo apt-get update
sudo apt install -y ca-certificates curl gnupg
```

### 3.2 添加 Docker 官方 GPG key
```zsh
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

### 3.3 添加 Docker 官方 APT 源
```zsh
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```
这一步的作用是把 Docker 官方软件源写入系统，这样后续`apt update`才能从官方仓库安装 Docker。

### 3.4 安装 Docker
```zsh
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 3.5 启动并设置开机自启
```zsh
sudo systemctl start docker
sudo systemctl enable docker
```

### 3.6 验证 Docker 是否安装成功
```zsh
docker --version
```

## 4. 给 Docker daemon 配置代理
> 这一部分非常重要。即使你当前 shell 已经能走代理，也不代表 Docker daemon 能走代理。
> Docker 拉镜像时是 dockerd 在联网，不是你当前 shell。

### 4.1 创建 systemd 配置目录
```zsh
sudo mkdir -p /etc/systemd/system/docker.service.d
```

### 4.2 写入代理配置
```zsh
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf > /dev/null <<'EOF'
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7890"
Environment="HTTPS_PROXY=http://127.0.0.1:7890"
Environment="NO_PROXY=localhost,127.0.0.1"
EOF
```
如果你的代理端口不是`7890`，需要改成自己的实际端口。

### 4.3 重载并重启 Docker
```zsh
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 4.4 检查代理是否生效
```zsh
systemctl show --property=Environment docker
```
如果配置成功，应该能看到类似：
```zsh
Environment=HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 ...
```

## 5 先用 root / sudo 测试 Docker
先测试 Docker 本身是否能正常运行：
```zsh
sudo docker run hello-world
```
这条命令的作用是：
- 如果本地没有`hello-world`镜像，就先从远程拉取
- 基于这个镜像启动一个最小容器
- 容器运行后打印测试信息并退出
如果看到类似：
```zsh
Hello from Docker!
```
说明 Docker 安装和 daemon 运行正常。

## 6 让普通用户也能直接使用 Docker
### 6.1 查看有哪些普通用户
```zsh
ls /home
```

### 6.2 把目标用户加入`docker`组
假设你的普通用户名是`xxx`：
```zsh
sudo usermod -aG docker xxx
```

### 6.3 检查是否真的加入成功
在 root 下执行：
```zsh
getent group docker
```
应该能看到类似：
```zsh
docker:x:999:xxx,root
```
如果这里已经有`xxx`，说明系统层面的组配置已经成功。

## 7. 验证普通用户是否能直接用 Docker
重新以普通用户登录服务器后，执行：
```zsh
id xiongxiao
id
groups
```
如果一切正常，`id`和`groups`都应该看到`docker`组。
然后执行：
```zsh
docker run hello-world
```
如果成功，就说明普通用户已经可以直接用 Docker。

## 8. 常见坑：用户已经加入 docker 组，但当前登录会话没有加载
我这里实际遇到过一个特殊问题：
```zsh
id xxx
```
显示：
```zsh
uid=1000(xxx) gid=1000(xxx) groups=1000(xxx),999(docker)
```
但当前会话执行：
```zsh
id
groups
```
却只显示：
```zsh
uid=1000(xxx) gid=1000(xxx) groups=1000(xxx)
xxx
```
这说明：
- 系统层面已经把用户加入`docker`组了
- 但当前 SSH / 终端会话没有正确加载 supplementary groups
- 导致当前会话里访问 /var/run/docker.sock 仍然会报权限错误
典型报错：
```zsh
permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
```

### 8.1 临时修复方法
如果当前 shell 没有加载`docker`组，可以先执行：
```zsh
newgrp docker
```
然后再执行：
```zsh
docker run hello-world
```
这通常能临时解决问题。但是每开一个新 shell 都要重新执行一次。

### 8.2 针对 zsh + Cursor Remote SSH 的自动修复
我当前环境里：
- shell 是 zsh
- 远程连接工具是 Cursor

所以为了让每次连接都自动获得 docker 组权限，可以分别配置：

#### 8.2.1 修改`~/.zprofile`
```zsh
code ~/.zprofile
```
添加：
```zsh
if ! id -nG | grep -qw docker && [ -z "$DOCKER_GROUP_FIXED" ]; then
  export DOCKER_GROUP_FIXED=1
  exec sg docker "exec zsh -l"
fi
```

#### 8.2.2 修改`~/.zshrc`
```zsh
code ~/.zshrc
```
添加：
```zsh
if ! id -nG | grep -qw docker && [ -z "$DOCKER_GROUP_FIXED" ]; then
  export DOCKER_GROUP_FIXED=1
  exec sg docker "exec zsh -i"
fi
```

### 8.3 修改后怎么验证
重新连接服务器, 登录后执行
```zsh
id
groups
```
理想情况应该能看到：
```zsh
groups=1000(xxx),999(docker)
```
然后再执行：
```zsh
docker run hello-world
```

## 9. 打通 Docker + NVIDIA GPU（Container Toolkit / CDI）
> 目标：让 Docker 容器可以正确访问宿主机 GPU
### 9.1 安装 NVIDIA Container Toolkit
```zsh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
```

### 9.2 配置 Docker 支持 GPU
```zsh
sudo nvidia-ctk runtime configure --runtime=docker --cdi.enabled=true
```

### 9.3 生成 CDI 配置文件
```zsh
sudo mkdir -p /etc/cdi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

### 9.4 重启 Docker
```zsh
sudo systemctl restart docker
```

### 9.5 验证 Docker 能否识别 GPU
```zsh
docker run --rm --gpus all ubuntu nvidia-smi
```

## 10. 从 4090 裸金属访问 Coder Workspace

> 目标：让本地 4090 裸金属机器作为 benchmark client，连接到远程 Coder Workspace 上运行的 model server。

### 10.1 安装 Coder CLI
```zsh
curl -L https://coder.com/install.sh | sh
```

### 10.2 登录 Coder
```zsh
coder login https://coder.lionrock.com/
```

### 10.3 配置 hosts 解析
由于 Coder 平台的域名可能需要手动解析，编辑 `/etc/hosts`：
```zsh
sudo code /etc/hosts
```
添加以下内容（IP 根据实际情况修改）：
```
106.13.249.94 coder.lionrock.com
106.13.249.94 8000--main--xiaoxiong-8gpu-0--xiaoxiong-sherry.coder.lionrock.com
```
> **前提**：需要先在 Coder Workspace（GPU 服务器）上启动 model server：
> ```zsh
> vla-eval serve --config configs/model_servers/dexbotic_cogact_libero.yaml
> ```
> 启动后才能在 Coder 面板上看到对应端口并获取地址。
>
> **获取地址的步骤**：
> 1. 打开 https://coder.lionrock.com/，找到目标 Workspace
> 2. 点击 **Open Ports** 按钮
> 3. 找到 model server 对应的端口
> 4. 点击 **Share this port**，将 Sharing Level 改为 **Public**
> 5. 右键复制链接地址

### 10.4 修改评测配置文件
编辑 `configs/libero_smoke_test.yaml`，将 `server.url` 指向 Coder Workspace 的 model server 地址：
```yaml
server:
  url: "wss://8000--main--xiaoxiong-8gpu-0--xiaoxiong-sherry.coder.lionrock.com"
```

### 10.5 修改连接代码

Coder Workspace 使用 `wss://`（TLS）连接，但其证书不被默认信任，需要在 `src/vla_eval/connection.py` 的 `_connect_with_backoff` 方法中添加 SSL 支持并跳过证书验证。

具体改动：

1. **第 7 行：新增 `import ssl`**

2. **第 192~202 行：修改 `_connect_with_backoff` 方法**，将原来直接传参的 `websockets.connect` 调用改为先构建 `kwargs` 字典，再根据 URL 协议决定是否注入 SSL 上下文：
```python
kwargs: dict[str, Any] = dict(
    compression=None,
    max_size=None,
    ping_interval=None,  # server may block GIL during JIT warmup
)
if self.url.startswith("wss://"):
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    kwargs["ssl"] = ssl_ctx
self._ws = await websockets.connect(self.url, **kwargs)
```

> **为什么要跳过证书验证**：Coder 平台暴露的端口使用的 TLS 证书通常是自签名的或不被系统 CA 信任，直接连接会报 `ssl.SSLCertVerificationError`。设置 `check_hostname=False` + `CERT_NONE` 可以绕过这个问题。

### 10.6 运行评测
完成以上配置后，在 4090 裸金属机器上执行：
```zsh
vla-eval run --dev --config configs/libero_smoke_test.yaml
```
> `--dev` 标志会把本地修改过的 `src/` 目录挂载到 Docker 容器的 `/workspace/src`，这样容器内会使用你本地的代码改动（如 10.5 中的 SSL 修改），而不需要重新构建镜像。
