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