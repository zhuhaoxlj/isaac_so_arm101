# VS Code Remote-SSH 连接 OpenBayes 云机器 —— 问题诊断与修复

> 环境：OpenBayes 云容器 / Ubuntu 22.04 LTS (x86_64) / VS Code 1.109.5

---

## 目标

在 Mac 上通过 VS Code Remote-SSH 连接 OpenBayes 云机器（`ssh.openbayes.com`），实现远程开发。

---

## 遇到的问题

### 问题一：初始化超时

**现象：**

```
Setting up SSH Host ssh.openbayes.com: Initializing VS Code Server
[Timeout]: Error: Timeout (Connecting with SSH timed out)
```

SSH 认证成功，但卡在 "Initializing VS Code Server" 阶段，最终超时。

**根因诊断：**

```bash
# 检查登录 Shell
cat /etc/passwd | grep root
# 输出：root:x:0:0:root:/root:/usr/bin/fish   ← 问题所在！
```

**根本原因**：登录 Shell 为 `fish`。VS Code Remote-SSH 初始化时通过 SSH 发送 POSIX/bash 语法的命令（如 `export VAR=value`），fish shell 无法解析这些命令，导致初始化脚本静默失败并超时。

**修复：**

```bash
# 将登录 Shell 改为 bash（不影响交互式使用 fish）
usermod -s /bin/bash root

# 验证
grep root /etc/passwd
# 期望输出：root:x:0:0:root:/root:/bin/bash  ✓
```

> 💡 若仍想在终端中使用 fish，可在 `~/.bashrc` 末尾添加 `exec fish`，bash 登录后自动切换到 fish，但不影响 VS Code Server 初始化。

---

### 问题二：一直卡在 "Downloading VS Code Server"

**现象：**

```
Setting up SSH Host ssh.openbayes.com: Downloading VS Code Server
```

解决 fish shell 问题后，VS Code 改为卡在下载阶段。

**诊断过程：**

```bash
# 1. 检查预安装的 Server 文件是否存在
ls -la /root/.vscode-server/bin/072586267e68ece9a47aa43f8c108e0dcbf44622/
# ✓ 文件存在，二进制可正常执行

# 2. 检查网络连通性
curl -fsSL --max-time 5 "https://update.code.visualstudio.com/..." -o /dev/null -w "%{http_code}"
# HTTP 200 ✓，网络正常

# 3. 发现关键线索 —— 目录内出现异常文件
ls /root/.vscode-server/bin/<commit>/vs.tar.gz          # 74MB，之前连接失败时 VS Code 下载的残留
ls /root/.vscode-server/vscode-cli-<commit>.tar.gz       # 0 字节！这才是当前卡住的原因
```

**根本原因**：VS Code 1.109.x 默认使用 **Exec Server 模式**（日志路径中带 `-es` 后缀），该模式需要下载 `vscode-cli`（VS Code CLI 工具），而非传统的 `vs.tar.gz` Server 包。

两者区别：

| 模式 | 文件 | 大小 | 我们预装的 |
|------|------|------|-----------|
| **传统 Server** | `~/.vscode-server/bin/<commit>/` | ~120MB | ✓ 已装 |
| **Exec Server（默认）** | `~/.vscode-server/vscode-cli-<commit>.tar.gz` | ~10MB | ✗ 缺失 → 卡住 |

**修复：**

```bash
COMMIT="072586267e68ece9a47aa43f8c108e0dcbf44622"

# 下载 VS Code CLI（Exec Server 所需）
curl -fsSL "https://update.code.visualstudio.com/commit:${COMMIT}/cli-alpine-x64/stable" \
  -o "/root/.vscode-server/vscode-cli-${COMMIT}.tar.gz"

# 验证
ls -lh /root/.vscode-server/vscode-cli-${COMMIT}.tar.gz
# 期望：9.6M  ✓
```

**备用方案（禁用 Exec Server 模式，改用传统模式）：**

在 Mac VS Code 的 `settings.json` 中添加：

```json
"remote.SSH.useExecServer": false
```

---

## 完整安装步骤（全新环境从头配置）

### 第一步：云机器配置

```bash
# 1. 确认 SSH 服务正常
ss -tlnp | grep :22

# 2. 修改登录 Shell 为 bash（必须）
usermod -s /bin/bash root

# 3. 预安装 VS Code Server（传统模式）
COMMIT="072586267e68ece9a47aa43f8c108e0dcbf44622"
INSTALL_DIR="$HOME/.vscode-server/bin/$COMMIT"
mkdir -p "$INSTALL_DIR"
curl -fsSL "https://update.code.visualstudio.com/commit:${COMMIT}/server-linux-x64/stable" \
  -o /tmp/vscode-server.tar.gz
tar -xzf /tmp/vscode-server.tar.gz -C "$INSTALL_DIR" --strip-components=1
touch "$INSTALL_DIR/0"    # 安装完成标记
rm -f /tmp/vscode-server.tar.gz

# 4. 预安装 VS Code CLI（Exec Server 模式，VS Code 1.100+ 默认使用）
curl -fsSL "https://update.code.visualstudio.com/commit:${COMMIT}/cli-alpine-x64/stable" \
  -o "/root/.vscode-server/vscode-cli-${COMMIT}.tar.gz"

# 5. 验证
/root/.vscode-server/bin/$COMMIT/bin/code-server --version
```

### 第二步：Mac 配置

```bash
# 生成 SSH 密钥（如果还没有）
ssh-keygen -t ed25519 -C "your@email.com"

# 上传公钥到云机器
ssh-copy-id -p 32090 root@ssh.openbayes.com
```

编辑 `~/.ssh/config`：

```
Host openbayes
    HostName ssh.openbayes.com
    User root
    Port 32090
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

### 第三步：VS Code 安装 Remote-SSH 插件

- 安装版本 1.109.x（与预安装的 Server commit 对应）
- `Cmd+Shift+X` 搜索 `Remote - SSH` 安装
- 左下角 `><` → **Connect to Host** → 选 `openbayes`

---

## ⚠️ 持久化警告

OpenBayes 容器的 `/root` 目录**在容器重启后会被重置**，导致 VS Code Server 丢失，需重新安装。

**持久化方案：**

```bash
# 将 .vscode-server 迁移到持久化存储目录
mv /root/.vscode-server /output/.vscode-server
ln -s /output/.vscode-server /root/.vscode-server

# 验证软链接
ls -la /root/.vscode-server
# 期望：/root/.vscode-server -> /output/.vscode-server  ✓
```

`/output` 目录挂载在持久化网络存储上，容器重启后数据不丢失。

---

## 问题速查表

| 现象 | 原因 | 修复 |
|------|------|------|
| Initializing VS Code Server 超时 | 登录 Shell 为 fish，不兼容 VS Code 初始化脚本 | `usermod -s /bin/bash root` |
| Downloading VS Code Server 卡住 | 缺少 `vscode-cli-<commit>.tar.gz`（Exec Server 模式需要） | 手动下载 CLI 包，或设置 `"remote.SSH.useExecServer": false` |
| 重启后再次出现相同问题 | `/root` 目录不持久化，Server 被重置 | 迁移到 `/output` 并创建软链接 |

---

*记录时间：2026-02-22*
