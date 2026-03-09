# HIL-SERL Learner 模块 Intel GPU 适配技术规格文档

文档版本: 1.0
日期: 2026-03-09
交付: Intel
项目仓库: https://github.com/rail-berkeley/hil-serl

---

## 目录

1. [项目背景与适配目标](#1-项目背景与适配目标)
2. [系统架构总览](#2-系统架构总览)
3. [Actor-Learner 通信协议详解](#3-actor-learner-通信协议详解)
4. [Learner 需要适配的完整文件清单与调用关系](#4-learner-需要适配的完整文件清单与调用关系)
5. [需要支持的 JAX 算子清单](#5-需要支持的-jax-算子清单)
6. [对外通信接口规格](#6-对外通信接口规格)
7. [技术修改建议](#7-技术修改建议)
8. [测试方案](#8-测试方案)
9. [附录：关键数据结构定义](#9-附录关键数据结构定义)

---

## 1. 项目背景与适配目标

### 1.1 项目简介

HIL-SERL（Human-in-the-Loop Sample-Efficient Reinforcement Learning）是一个机器人强化学习框架，通过人类示教（20 条轨迹）+ 人工干预纠错 + 奖励分类器 + SAC 在线强化学习的组合，在 1.5-2.5 小时内将 Franka 机械臂操作任务训练到 90-100% 成功率。

### 1.2 适配目标

将 **Learner 进程**（训练算法模块）部署到 Intel CPU+GPU 设备上。Learner 是项目中 GPU 计算量最大的模块（占总 GPU 工作量 ~90%），负责：

- 从 Replay Buffer 中采样 batch
- 执行数据增强（随机裁剪）
- 计算 Critic/Actor/Temperature 的梯度并更新参数
- 维护 Target Network（Polyak 平均）
- 将更新后的参数发布给 Actor

### 1.3 适配边界

| 模块 | 运行位置 | 适配方 |
|------|---------|--------|
| **Learner（训练算法）** | **Intel 设备** | **Intel** |
| Actor（策略推理+环境交互） | 原有 NVIDIA 机器 | 不改动 |
| 机器人控制（ROS/Flask） | 原有机器 | 不改动 |
| 传感器（RealSense/SpaceMouse） | 原有机器 | 不改动 |

### 1.4 当前技术栈

| 组件 | 版本 |
|------|------|
| ML 框架 | JAX（通过 `jax[cuda12_pip]==0.4.35`） |
| 神经网络 | Flax >= 0.8.0 |
| 优化器 | Optax >= 0.1.5 |
| 概率分布 | Distrax >= 0.1.2 |
| 通信框架 | agentlace 0.1.4（ZeroMQ） |
| 序列化 | pickle + LZ4 压缩 |

---

## 2. 系统架构总览

### 2.1 适配后的部署架构

```
┌──────────────────────────────────────┐       ┌──────────────────────────────────────┐
│     原有 NVIDIA 机器（连接机器人）      │       │         Intel CPU+GPU 设备            │
│                                      │       │                                      │
│  ┌────────────────────────────────┐  │       │  ┌────────────────────────────────┐  │
│  │        Actor 进程              │  │       │  │        Learner 进程             │  │
│  │                                │  │       │  │                                │  │
│  │  环境交互 (FrankaEnv)          │  │       │  │  TrainerServer                 │  │
│  │  ├─ RealSense 相机采集         │  │       │  │  ├─ REQ-REP (port 5588)        │  │
│  │  ├─ SpaceMouse 人工干预        │  │       │  │  └─ PUB-SUB (port 5589)        │  │
│  │  └─ Flask HTTP → 机器人控制    │  │       │  │                                │  │
│  │                                │  │       │  │  Replay Buffer (NumPy/CPU)     │  │
│  │  策略推理 (GPU 轻量)           │  │       │  │  ├─ online_buffer              │  │
│  │  └─ agent.sample_actions()    │  │       │  │  └─ demo_buffer                │  │
│  │                                │  │       │  │                                │  │
│  │  TrainerClient                 │  │  ZMQ  │  │  训练循环 (GPU 密集)            │  │
│  │  ├─ QueuedDataStore ──────────────────────►│  │  ├─ batch 采样 + 增强          │  │
│  │  │  (发送 transitions)         │  │  :5588│  │  ├─ agent.update() [JIT]       │  │
│  │  │                             │  │       │  │  │  ├─ critic_loss → grad      │  │
│  │  │                             │  │       │  │  │  ├─ actor_loss → grad       │  │
│  │  │                             │  │       │  │  │  └─ temperature_loss → grad │  │
│  │  └─ recv_network_callback ◄───────────────│  │  ├─ target_update (Polyak)     │  │
│  │     (接收新参数)               │  │  :5589│  │  └─ publish_network(params)    │  │
│  │                                │  │       │  │                                │  │
│  │  send-stats ──────────────────────────────►│  │  WandB 日志                    │  │
│  │  (发送环境统计)                │  │  :5588│  │  Checkpoint 保存               │  │
│  └────────────────────────────────┘  │       │  └────────────────────────────────┘  │
└──────────────────────────────────────┘       └──────────────────────────────────────┘
```

### 2.2 Learner 进程生命周期

```
启动 → 加载配置 → 创建 fake_env(仅用于获取观测/动作空间维度)
     → 创建 SAC Agent（初始化网络参数，加载 ResNet-10 预训练权重）
     → 加载 demo 数据到 demo_buffer
     → 启动 TrainerServer（监听 port 5588/5589）
     → 等待 Actor 填充 replay_buffer 至 training_starts 条数据
     → 进入训练循环：
         每步：
           1. 从 replay_buffer 采样 batch_size/2 条
           2. 从 demo_buffer 采样 batch_size/2 条
           3. 拼接为完整 batch
           4. 执行 agent.update()（JIT 编译，GPU 上执行）
           5. 每 steps_per_update 步：publish_network(params)
           6. 每 log_period 步：写入 WandB
           7. 每 checkpoint_period 步：保存 checkpoint
```

---

## 3. Actor-Learner 通信协议详解

### 3.1 通信框架

使用 **agentlace** 库（MIT 协议，https://github.com/youliangtan/agentlace ），基于 ZeroMQ 实现。

### 3.2 网络拓扑

| 通道 | ZMQ 模式 | 默认端口 | 方向 | 用途 |
|------|---------|---------|------|------|
| 数据通道 | REQ-REP | **5588** | Actor → Learner | 传输 transitions、统计信息 |
| 参数广播 | PUB-SUB | **5589** | Learner → Actor(s) | 广播更新后的网络参数 |

### 3.3 序列化协议

所有消息均通过以下流程序列化：

```
Python dict → pickle.dumps() → lz4.frame.compress() → ZMQ socket.send()
ZMQ socket.recv() → lz4.frame.decompress() → pickle.loads() → Python dict
```

### 3.4 消息类型定义

#### 3.4.1 数据传输消息（Actor → Learner，REQ-REP port 5588）

**请求格式**：
```python
{
    "type": "datastore",
    "store_name": "actor_env",        # 或 "actor_env_intvn"（干预数据）
    "payload": {
        "data": [transition_1, transition_2, ...],  # 新增的 transitions 列表
        "last_id": 1234                              # 客户端序列号
    }
}
```

**Transition 数据结构**（每条 transition）：
```python
{
    "observations": {
        "state": np.ndarray(shape=(1, 17), dtype=float32),
        "wrist_1": np.ndarray(shape=(1, 128, 128, 3), dtype=uint8),
        "wrist_2": np.ndarray(shape=(1, 128, 128, 3), dtype=uint8),
    },
    "actions": np.ndarray(shape=(7,), dtype=float32),     # [-1, 1]
    "next_observations": {
        "state": np.ndarray(shape=(1, 17), dtype=float32),
        "wrist_1": np.ndarray(shape=(1, 128, 128, 3), dtype=uint8),
        "wrist_2": np.ndarray(shape=(1, 128, 128, 3), dtype=uint8),
    },
    "rewards": float32,          # 0.0 或 1.0
    "masks": float32,            # 1.0 - done
    "dones": bool,               # True/False
    "grasp_penalty": float32,    # 可选，-0.02 或 0.0
}
```

**state 向量（17维）组成**：
```
[0]      gripper_pose     (1)  夹爪位置
[1:7]    tcp_pose         (6)  末端位姿 [x, y, z, roll, pitch, yaw]
[7:13]   tcp_vel          (6)  末端速度 [vx, vy, vz, wx, wy, wz]
[13:16]  tcp_force        (3)  末端力 [fx, fy, fz]
[16:19]  tcp_torque       (3)  末端力矩 [tx, ty, tz]
```

**响应格式**：
```python
{"success": True}
```

#### 3.4.2 统计信息消息（Actor → Learner，REQ-REP port 5588）

**请求格式**：
```python
{
    "type": "send-stats",
    "payload": {
        # 类型 1：环境统计（每个 episode 结束时发送）
        "environment": {
            "episode": {
                "r": float,                    # episode 总回报
                "l": int,                      # episode 长度
                "t": float,                    # episode 耗时(秒)
            },
            "intervention_count": int,         # 人工干预次数
            "intervention_steps": int,         # 人工干预步数
        },
        # 类型 2：计时统计（每 log_period 步发送）
        "timer": {
            "sample_actions": float,           # 推理耗时(秒)
            "step_env": float,                 # 环境交互耗时(秒)
            "total": float,                    # 总耗时(秒)
        }
    }
}
```

**响应格式**：
```python
{}  # 空 dict
```

#### 3.4.3 网络参数广播（Learner → Actor(s)，PUB-SUB port 5589）

**广播内容**：`agent.state.params` — 一个 JAX PyTree（嵌套 dict 结构，叶子节点为 `jnp.ndarray`）

**PyTree 结构**：
```python
{
    "modules_actor": {
        "actor": {
            "Dense_0": {"kernel": jnp.array(...), "bias": jnp.array(...)},
            "Dense_1": {"kernel": jnp.array(...), "bias": jnp.array(...)},
            ...
        },
        "encoder": {
            "encoder_wrist_1": {
                "pretrained_encoder": {
                    "ResNetBlock_0": {...},
                    "ResNetBlock_1": {...},
                    ...
                },
                "SpatialLearnedEmbeddings_0": {...},
                "Dense_0": {...},
            },
            "encoder_wrist_2": { ... },  # 同上结构
        },
    },
    "modules_critic": {
        "critic": {
            "critic_ensemble": {
                "VmapCritic_0": {
                    "Dense_0": {...},
                    "Dense_1": {...},
                    ...
                },
            },
        },
        "encoder": { ... },  # 与 actor encoder 共享参数定义
    },
    "modules_temperature": {
        "temperature": {
            "value": jnp.array(scalar),  # 标量温度参数
        },
    },
}
```

**广播时机**：每 `steps_per_update`（默认 50）个训练步广播一次。

**广播机制**：
- ZMQ PUB socket 设置 `SNDHWM=3`（发送队列最多 3 条）
- ZMQ SUB socket 设置 `CONFLATE=True`（只保留最新一条），`RCVTIMEO=1500ms`
- Actor 在后台守护线程中接收，调用 `update_params(params)` 回调

#### 3.4.4 配置兼容性校验（连接建立时）

Actor 连接时首先发送 hash 校验请求：
```python
# 请求
{"type": "hash"}
# 响应
{"success": True, "payload": "<TrainerConfig 的 JSON 序列化>"}
```
客户端对比 MD5 hash，不匹配则拒绝连接。

#### 3.4.5 数据同步查询

```python
# 请求
{"type": "get_last_update_id", "payload": {"store_name": "actor_env"}}
# 响应
{"success": True, "payload": 1234}  # 服务端最新数据 ID
```

### 3.5 TrainerConfig 配置

```python
# 源码位置：serl_launcher/serl_launcher/utils/launcher.py:233-238
TrainerConfig(
    port_number=5588,           # REQ-REP 端口
    broadcast_port=5589,        # PUB-SUB 端口
    request_types=["send-stats"],  # 自定义请求类型
)
```

### 3.6 通信时序图

```
Actor (NVIDIA机器)                          Learner (Intel设备)
    |                                           |
    |──── hash 校验 ────────────────────────────>|
    |<─── config JSON ──────────────────────────|
    |                                           |
    |  [Actor 开始采集数据]                      |  [Learner 等待数据]
    |                                           |
    |──── datastore(transitions) ──────────────>|
    |<─── {"success": true} ───────────────────|
    |                                           |
    |  ... (多次数据传输) ...                    |  [replay_buffer 填充中]
    |                                           |
    |                                           |  [buffer >= training_starts]
    |                                           |  [开始训练循环]
    |                                           |
    |<──── publish_network(params) ────────────|  [PUB-SUB 广播]
    |  [更新本地策略参数]                        |
    |                                           |
    |──── send-stats(环境统计) ────────────────>|
    |<─── {} ──────────────────────────────────|  [写入 WandB]
    |                                           |
    |  [继续采集]                               |  [继续训练]
    |──── datastore(新 transitions) ───────────>|
    |                                           |
    |<──── publish_network(更新的 params) ─────|  [每50步广播]
    |                                           |
    |  ... (循环) ...                           |  ... (循环) ...
```

---

## 4. Learner 需要适配的完整文件清单与调用关系

### 4.1 文件依赖总览

以下是 Learner 进程运行时涉及的**全部**项目源文件。Intel 适配团队需要确保这些文件在 Intel 设备上能正确运行。

```
入口文件
└── examples/train_rlpd.py                      ← Learner 主入口

配置文件
├── examples/experiments/config.py               ← DefaultTrainingConfig 基类
├── examples/experiments/mappings.py             ← 实验名称→配置类映射
└── examples/experiments/ram_insertion/config.py  ← 任务配置（含 get_environment）

Agent 实现（核心 GPU 计算）
├── serl_launcher/agents/continuous/sac.py                    ← SACAgent 主类
├── serl_launcher/agents/continuous/sac_hybrid_single.py      ← 单臂混合动作 SAC
├── serl_launcher/agents/continuous/sac_hybrid_dual.py        ← 双臂混合动作 SAC
└── serl_launcher/agents/continuous/bc.py                     ← BCAgent（BC 训练用）

网络定义（GPU 前向/反向传播）
├── serl_launcher/networks/actor_critic_nets.py  ← Policy, Critic, GraspCritic, ensemblize
├── serl_launcher/networks/mlp.py                ← MLP, MLPResNet, Scalar
├── serl_launcher/networks/lagrange.py           ← GeqLagrangeMultiplier（温度参数）
├── serl_launcher/networks/reward_classifier.py  ← 奖励分类器（get_environment 中使用）
└── serl_launcher/networks/classifier.py         ← HumanClassifier

视觉编码器（GPU 密集计算）
├── serl_launcher/vision/resnet_v1.py            ← ResNet-10 编码器
├── serl_launcher/vision/data_augmentations.py   ← 随机裁剪、高斯模糊、颜色增强
├── serl_launcher/vision/film_conditioning_layer.py ← FiLM 条件层
└── serl_launcher/vision/spatial.py              ← 空间注意力

公共模块
├── serl_launcher/common/common.py       ← JaxRLTrainState, ModuleDict, shard_batch
├── serl_launcher/common/encoding.py     ← EncodingWrapper（多模态编码）
├── serl_launcher/common/optimizers.py   ← make_optimizer（optax 包装）
├── serl_launcher/common/typing.py       ← 类型定义
└── serl_launcher/common/wandb.py        ← WandB 日志

数据管理
├── serl_launcher/data/replay_buffer.py               ← ReplayBuffer 基类
├── serl_launcher/data/memory_efficient_replay_buffer.py ← 内存高效 Buffer
├── serl_launcher/data/data_store.py                   ← DataStore 适配器
└── serl_launcher/data/dataset.py                      ← Dataset 基类

工具函数
├── serl_launcher/utils/launcher.py      ← Agent 工厂、TrainerConfig、数据增强函数
├── serl_launcher/utils/train_utils.py   ← concat_batches, _unpack, load_resnet10_params
├── serl_launcher/utils/timer_utils.py   ← Timer 计时
├── serl_launcher/utils/jax_utils.py     ← JaxRNG, device_put 工具
└── serl_launcher/utils/logging_utils.py ← 日志工具

环境 Wrapper（Learner 仅在 fake_env=True 模式下使用）
├── serl_launcher/wrappers/serl_obs_wrappers.py ← 观测空间处理
├── serl_launcher/wrappers/chunking.py          ← 时间堆叠
├── franka_env/envs/franka_env.py               ← FrankaEnv（fake_env 模式）
├── franka_env/envs/wrappers.py                 ← 环境 wrapper
├── franka_env/envs/relative_env.py             ← 相对坐标变换
└── examples/experiments/ram_insertion/wrapper.py ← 任务 wrapper
```

### 4.2 核心调用链（训练一步的完整路径）

```
train_rlpd.py:learner()
│
├── replay_buffer.get_iterator()
│   └── replay_buffer.sample()                      [CPU: NumPy]
│       └── _device_put_batch(data, device)         [CPU→GPU 传输]
│           └── jax.device_put()                    ★ 需要 Intel GPU 支持
│
├── concat_batches(batch, demo_batch)               [GPU: jnp.concatenate]
│   └── train_utils.py:concat_batches()             ★ 需要 Intel GPU 支持
│
└── agent.update(batch, networks_to_update=...)     [GPU: JIT 编译执行]
    │  源码: sac.py:259-325
    │
    ├── _unpack(batch)                              [GPU: jnp 切片操作]
    │   └── train_utils.py:_unpack()
    │
    ├── augmentation_function(batch, rng)            [GPU: JIT]
    │   └── launcher.py:augment_batch()
    │       └── batched_random_crop()               [GPU: jax.vmap + dynamic_slice]
    │           └── data_augmentations.py:random_crop()
    │
    ├── self.loss_fns(batch)                        [构造损失函数闭包]
    │   ├── critic_loss_fn(batch, params, rng)      [GPU: 前向+损失计算]
    │   │   ├── forward_policy(next_obs)            → Policy 网络前向
    │   │   │   └── EncodingWrapper → ResNet-10 → MLP → Distribution
    │   │   ├── forward_target_critic(next_obs)     → Target Critic 前向
    │   │   │   └── EncodingWrapper → ResNet-10 → MLP → Q-value
    │   │   ├── forward_critic(obs, actions)        → Critic 前向
    │   │   └── MSE loss 计算
    │   │
    │   ├── policy_loss_fn(batch, params, rng)      [GPU: 前向+损失计算]
    │   │   ├── forward_policy(obs)                 → Policy 前向
    │   │   ├── forward_critic(obs, sampled_actions) → Critic 前向
    │   │   └── actor_loss = -(Q - α·log_π)
    │   │
    │   └── temperature_loss_fn(batch, params, rng) [GPU: 标量计算]
    │       └── Lagrange penalty 计算
    │
    ├── state.apply_loss_fns(loss_fns)              [GPU: 自动微分+梯度更新]
    │   └── common.py:JaxRLTrainState.apply_loss_fns()
    │       ├── jax.grad(loss_fn)(params, rng)      ★ 核心：自动微分
    │       ├── self.apply_gradients(grads)          ★ 核心：参数更新
    │       │   └── optax.adam.update()
    │       │   └── optax.apply_updates()
    │       └── jax.lax.pmean()                     [多设备梯度平均，可选]
    │
    └── state.target_update(tau=0.005)              [GPU: Polyak 平均]
        └── jax.tree_map(lambda p, tp: p*τ + tp*(1-τ), ...)
```

### 4.3 每个文件的修改必要性分析

| 文件 | GPU依赖 | 是否需修改 | 修改原因 |
|------|---------|-----------|---------|
| `train_rlpd.py` | 是（设备检测） | **可能** | L58-61: `jax.local_devices()` — 需确认 Intel GPU 设备被正确检测 |
| `sac.py` | 是（JIT编译） | **否** | 纯 JAX 代码，设备无关 |
| `common.py` | 是（device_put, grad） | **否** | 通过 JAX API 抽象，不直接调用 CUDA |
| `replay_buffer.py` | 是（device_put） | **可能** | L36-57: `_device_put_batch()` — 需确认 Intel sharding 兼容性 |
| `data_augmentations.py` | 是（JIT, vmap） | **否** | 纯 JAX lax 操作 |
| `resnet_v1.py` | 是（Conv, Dense） | **否** | 标准 Flax 层 |
| `actor_critic_nets.py` | 是（vmap, Dense） | **否** | 标准 Flax 层 + jax.vmap |
| `launcher.py` | 是（PRNGKey） | **否** | TrainerConfig 不涉及 GPU |
| `train_utils.py` | 是（concatenate） | **否** | jnp.concatenate 是标准操作 |
| `data_store.py` | 否（线程锁） | **否** | 纯 Python 线程同步 |
| `run_learner.sh` | 是（环境变量） | **是** | 需要调整 XLA 环境变量以适配 Intel GPU |

---

## 5. 需要支持的 JAX 算子清单

### 5.1 优先级 P0：核心训练算子（必须支持，否则无法训练）

| JAX 操作 | 用途 | 调用位置 |
|----------|------|---------|
| `jax.jit` | JIT 编译训练步骤 | `sac.py:259,327` |
| `jax.grad` | 计算损失函数梯度 | `common.py:205` |
| `jax.value_and_grad` | 同时获取损失值和梯度 | `common.py:204` (间接) |
| `jax.device_put` | CPU→GPU 数据传输 | `replay_buffer.py:38-57`, `train_rlpd.py:421-423` |
| `jax.device_get` | GPU→CPU 数据取回 | `train_rlpd.py:345` (block_until_ready) |
| `jax.tree_map` | PyTree 参数操作 | `common.py:25-30,119,132,162,204` |
| `jax.tree_util.tree_map` | 同上 | 多处 |
| `jax.tree_util.tree_structure` | PyTree 结构分析 | `common.py:199` |
| `jax.tree_util.tree_unflatten` | PyTree 重构 | `common.py:201` |
| `jax.tree_util.tree_leaves` | PyTree 叶子提取 | `train_utils.py:145` |
| `jax.random.split` | PRNG 密钥分割 | `sac.py:152,213,240`, 多处 |
| `jax.random.PRNGKey` | 创建 PRNG 密钥 | `train_rlpd.py:371`, `launcher.py:61` |
| `jax.random.randint` | 随机整数（裁剪偏移） | `data_augmentations.py:8` |
| `jax.block_until_ready` | 等待计算完成 | `train_rlpd.py:345` |

### 5.2 优先级 P0：Flax 层操作（必须支持，网络核心）

| Flax 操作 | 用途 | 调用位置 |
|-----------|------|---------|
| `flax.linen.Dense` | 全连接层 | `mlp.py`, `actor_critic_nets.py` |
| `flax.linen.Conv` | 卷积层 | `resnet_v1.py` |
| `flax.linen.LayerNorm` | 层归一化 | `mlp.py` |
| `flax.linen.GroupNorm` | 组归一化 | `resnet_v1.py` |
| `flax.linen.Dropout` | Dropout | `mlp.py` |
| `flax.linen.relu` / `tanh` / `swish` | 激活函数 | 多处 |
| `flax.linen.Module.init` | 参数初始化 | `sac.py:396` |
| `flax.linen.Module.apply` | 前向传播 | `common.py:109` (via apply_fn) |
| `flax.struct.PyTreeNode` | PyTree 节点定义 | `common.py:81`, `sac.py:21` |

### 5.3 优先级 P0：Optax 优化器操作

| Optax 操作 | 用途 | 调用位置 |
|------------|------|---------|
| `optax.adam` | Adam 优化器 | `optimizers.py` |
| `optax.GradientTransformation.init` | 初始化优化器状态 | `common.py:244` |
| `optax.GradientTransformation.update` | 计算参数更新 | `common.py:144` |
| `optax.apply_updates` | 应用参数更新 | `common.py:165` |
| `optax.chain` | 组合优化器变换 | `optimizers.py` |
| `optax.clip_by_global_norm` | 梯度裁剪 | `optimizers.py` |

### 5.4 优先级 P1：数据增强与分布操作

| JAX/库 操作 | 用途 | 调用位置 |
|-------------|------|---------|
| `jax.vmap` | 批量向量化（Ensemble Critic, 裁剪） | `actor_critic_nets.py:32`, `data_augmentations.py:30` |
| `jax.lax.dynamic_slice` | 随机裁剪 | `data_augmentations.py:19` |
| `jax.lax.conv_general_dilated` | 高斯模糊 | `data_augmentations.py:60-67` |
| `jax.lax.cond` | 条件执行（增强概率） | `data_augmentations.py:47,267,301` |
| `jax.lax.stop_gradient` | 梯度截断 | `encoding.py` (frozen encoder) |
| `jax.image.resize` | 图像缩放 | `data_augmentations.py:42` |
| `jnp.pad` | 图像填充 | `data_augmentations.py:10-16` |
| `jnp.concatenate` | 批次拼接 | `train_utils.py:63` |
| `distrax.MultivariateNormalDiag` | 策略动作分布 | `actor_critic_nets.py` |
| `distrax.Transformed` | Tanh 变换分布 | `actor_critic_nets.py` |
| `distrax.Distribution.sample_and_log_prob` | 动作采样+对数概率 | `sac.py:143,217` |

### 5.5 优先级 P2：多设备与辅助操作

| 操作 | 用途 | 调用位置 |
|------|------|---------|
| `jax.local_devices()` | 检测可用设备 | `train_rlpd.py:58` |
| `jax.sharding.PositionalSharding` | 多 GPU 分片 | `train_rlpd.py:60` |
| `jax.lax.pmean` | 多设备梯度平均 | `common.py:215`（当前单 GPU 未使用） |
| `jax.random.categorical` | 分类采样（Critic 子采样） | `sac.py:167` |
| `jax.random.uniform` | 均匀随机（颜色增强） | `data_augmentations.py` |
| `jax.random.permutation` | 随机排列 | `data_augmentations.py:286` |

---

## 6. 对外通信接口规格

Intel 设备上的 Learner 需要开放以下网络接口供原有 NVIDIA 机器上的 Actor 连接。

### 6.1 必须开放的端口

| 端口 | 协议 | 方向 | 说明 |
|------|------|------|------|
| **5588/TCP** | ZMQ REQ-REP | 入站（Actor→Learner） | 接收 transitions 数据和统计信息 |
| **5589/TCP** | ZMQ PUB-SUB | 出站（Learner→Actor） | 广播网络参数更新 |

### 6.2 Actor 侧的配置修改

Actor 启动时只需将 `--ip` 参数从 `localhost` 改为 Intel 设备的 IP 地址：

```bash
# 原来（同机部署）
python train_rlpd.py --actor --ip=localhost

# 适配后（跨机部署）
python train_rlpd.py --actor --ip=<INTEL_DEVICE_IP>
```

对应代码位置：`train_rlpd.py:46`
```python
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
```

### 6.3 数据接口契约

Intel Learner 必须保证以下接口行为与原实现完全一致：

| 接口 | 契约 |
|------|------|
| 接收 `datastore` 消息 | 正确反序列化 transition dict，插入对应的 replay_buffer |
| 接收 `send-stats` 消息 | 正确解析统计 dict，返回空 dict `{}` |
| 接收 `hash` 消息 | 返回与 Actor 相同的 `TrainerConfig` JSON 序列化 |
| 接收 `get_last_update_id` 消息 | 返回对应 datastore 的最新序列号 |
| 广播 `publish_network` | 发布的 params PyTree 结构必须与 Agent 初始化时一致 |

### 6.4 网络性能要求

| 指标 | 要求 | 说明 |
|------|------|------|
| 网络延迟 | < 50ms | Actor-Learner 间的 RTT |
| 带宽 | > 100Mbps | 参数广播约 50-100MB/次 |
| 参数广播频率 | 每 50 训练步一次 | 约 1-5 秒一次 |
| 数据上传频率 | 每 episode 结束时批量上传 | 约 10-15 秒一次 |

---

## 7. 技术修改建议

### 7.1 环境配置修改

#### 7.1.1 `run_learner.sh` — 必须修改

**文件位置**：`examples/experiments/ram_insertion/run_learner.sh`

**当前内容**：
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=ram_insertion \
    --checkpoint_path=first_run \
    --demo_path=demo_data/xxx.pkl \
    --learner \
```

**建议修改**：
```bash
# Intel GPU 环境变量
export JAX_PLATFORMS=xpu                        # 指定使用 Intel GPU 后端
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # 保留：不预分配显存
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3        # 保留：限制显存占用

# 如果需要额外的 Intel oneAPI 环境
# source /opt/intel/oneapi/setvars.sh

python ../../train_rlpd.py "$@" \
    --exp_name=ram_insertion \
    --checkpoint_path=first_run \
    --demo_path=demo_data/xxx.pkl \
    --learner \
```

**修改原因**：
- `JAX_PLATFORMS=xpu`：告诉 JAX 使用 Intel GPU（XPU）后端而非 CUDA
- 原有的 `XLA_PYTHON_CLIENT_*` 变量在 Intel XLA 后端中可能仍然有效，需验证
- 可能需要 source oneAPI 环境变量

#### 7.1.2 设备检测代码 — 可能需修改

**文件位置**：`examples/train_rlpd.py` 第 58-61 行

**当前代码**：
```python
devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)
iterator_device = devices[0] if num_devices == 1 else sharding.replicate()
```

**分析**：此代码已经是设备无关的（`jax.local_devices()` 会自动检测当前后端可用的设备）。如果 `intel-extension-for-openxla` 正确注册了 PJRT 插件，`jax.local_devices()` 应返回 Intel GPU 设备。

**可能需要的修改**：如果 Intel GPU 返回多个子设备（tiles），需要确认 `PositionalSharding` 的行为是否正确。建议添加防御性代码：

```python
devices = jax.local_devices()
num_devices = len(devices)
print(f"Detected {num_devices} device(s): {devices}")  # 调试输出
sharding = jax.sharding.PositionalSharding(devices)
iterator_device = devices[0] if num_devices == 1 else sharding.replicate()
```

#### 7.1.3 `_device_put_batch` — 可能需修改

**文件位置**：`serl_launcher/serl_launcher/data/replay_buffer.py` 第 36-57 行

**当前代码**：
```python
def _device_put_batch(data: DatasetDict, device):
    if device is None:
        return jax.device_put(data)
    if isinstance(device, jax.sharding.Sharding):
        def put_leaf(x):
            if hasattr(x, "ndim") and hasattr(device, "reshape") and x.ndim >= 1:
                try:
                    sharding = device.reshape(device.shape[0], *((1,) * (x.ndim - 1)))
                    return jax.device_put(x, device=sharding)
                except ValueError:
                    return jax.device_put(x)
            try:
                return jax.device_put(x, device=device)
            except ValueError:
                return jax.device_put(x)
        return jax.tree_util.tree_map(put_leaf, data)
    return jax.device_put(data, device=device)
```

**分析**：此代码已经有 try-except 容错逻辑。但如果 Intel GPU 的 sharding 行为与 NVIDIA 不同，可能需要调整 reshape 逻辑。

### 7.2 Python 依赖安装

Intel 设备上需要安装以下依赖：

```bash
# 1. JAX Intel GPU 版
pip install jax jaxlib
pip install intel-extension-for-openxla    # Intel GPU PJRT 插件

# 2. 或者 JAX CPU 版（备选方案）
# pip install jax jaxlib

# 3. ML 依赖
pip install flax>=0.8.0
pip install optax>=0.1.5
pip install distrax>=0.1.2
pip install chex>=0.1.85

# 4. 通信框架
pip install agentlace@git+https://github.com/youliangtan/agentlace.git@cf2c337c5e3694cdbfc14831b239bd657bc4894d

# 5. 数据与工具依赖
pip install numpy>=1.24.3
pip install scipy==1.11.4
pip install gymnasium==0.29.1
pip install einops>=0.6.1
pip install wandb>=0.12.14
pip install ml_collections>=0.1.0
pip install absl-py>=0.12.0
pip install tqdm>=4.60.0
pip install imageio>=2.31.1
pip install opencv-python
pip install lz4
pip install pyzmq
pip install requests
pip install natsort
pip install pynput

# 6. TensorFlow（仅用于 train_utils.py 中的视频加载，可选）
pip install tensorflow>=2.15.0

# 7. 安装项目包
cd hil-serl/serl_launcher && pip install -e .
cd hil-serl/serl_robot_infra && pip install -e .
```

### 7.3 ResNet-10 预训练权重

Learner 初始化时会下载 ResNet-10 预训练权重（约 40MB）：

- URL: `https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl`
- 本地缓存: `~/.serl/resnet10_params.pkl`
- 格式: pickle 序列化的 Python dict（NumPy 数组）
- 代码位置: `train_utils.py:103-164`

如果 Intel 设备无法访问外网，可预先下载后放置到 `~/.serl/resnet10_params.pkl`。

### 7.4 Demo 数据传输

Learner 启动时需要加载 demo 数据文件（pickle 格式）：

```bash
# 从 NVIDIA 机器拷贝 demo 数据到 Intel 设备
scp nvidia-machine:/path/to/demo_data/*.pkl intel-device:/path/to/demo_data/
```

Demo 文件内容：list of transition dicts（结构同 3.4.1 节）。

---

## 8. 测试方案

### 8.1 阶段一：JAX 基础算子验证

**目标**：确认 JAX 在 Intel GPU 上能正确执行所有需要的操作。

**测试脚本** `test_01_jax_basics.py`：

```python
"""阶段一：JAX 基础算子验证（在 Intel 设备上运行）"""
import jax
import jax.numpy as jnp
import sys

def test_device_detection():
    """测试 1.1：设备检测"""
    devices = jax.local_devices()
    print(f"Devices: {devices}")
    print(f"Default backend: {jax.default_backend()}")
    assert len(devices) >= 1, "No devices found"
    print("PASS: Device detection")

def test_jit_matmul():
    """测试 1.2：JIT 矩阵乘法"""
    x = jnp.ones((256, 256))
    f = jax.jit(lambda x: x @ x)
    result = f(x)
    assert result.shape == (256, 256)
    assert jnp.allclose(result, jnp.full((256, 256), 256.0))
    print("PASS: JIT matmul")

def test_grad():
    """测试 1.3：自动微分"""
    def loss_fn(params):
        return jnp.mean(params ** 2)
    grad_fn = jax.jit(jax.grad(loss_fn))
    params = jnp.ones((64,))
    grads = grad_fn(params)
    assert grads.shape == (64,)
    assert jnp.allclose(grads, 2.0 / 64.0)
    print("PASS: Gradient computation")

def test_vmap():
    """测试 1.4：向量化映射"""
    def single_fn(x):
        return jnp.sum(x ** 2)
    batched_fn = jax.vmap(single_fn)
    x = jnp.ones((8, 32))
    result = batched_fn(x)
    assert result.shape == (8,)
    print("PASS: vmap")

def test_random():
    """测试 1.5：随机数生成"""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (64,))
    idx = jax.random.randint(k2, (10,), 0, 100)
    assert x.shape == (64,)
    assert idx.shape == (10,)
    print("PASS: Random number generation")

def test_device_put_get():
    """测试 1.6：CPU-GPU 数据传输"""
    import numpy as np
    x_np = np.random.randn(128, 128, 3).astype(np.float32)
    x_jax = jax.device_put(x_np)
    x_back = jax.device_get(x_jax)
    assert np.allclose(x_np, x_back, atol=1e-6)
    print("PASS: device_put/device_get")

def test_dynamic_slice():
    """测试 1.7：动态切片（数据增强核心）"""
    img = jnp.ones((128, 128, 3))
    padded = jnp.pad(img, ((4, 4), (4, 4), (0, 0)), mode="edge")
    crop_from = jnp.array([2, 3, 0], dtype=jnp.int32)
    cropped = jax.lax.dynamic_slice(padded, crop_from, (128, 128, 3))
    assert cropped.shape == (128, 128, 3)
    print("PASS: dynamic_slice")

def test_conv():
    """测试 1.8：卷积操作（ResNet 核心）"""
    x = jnp.ones((1, 64, 64, 3))
    kernel = jnp.ones((3, 3, 3, 32))
    result = jax.lax.conv_general_dilated(
        x, kernel, (1, 1), "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    assert result.shape == (1, 64, 64, 32)
    print("PASS: conv_general_dilated")

def test_tree_operations():
    """测试 1.9：PyTree 操作"""
    tree = {"a": jnp.ones(3), "b": {"c": jnp.zeros(5)}}
    doubled = jax.tree_map(lambda x: x * 2, tree)
    assert jnp.allclose(doubled["a"], jnp.full(3, 2.0))
    assert jnp.allclose(doubled["b"]["c"], jnp.zeros(5))
    leaves = jax.tree_util.tree_leaves(tree)
    assert len(leaves) == 2
    print("PASS: tree operations")

def test_sharding():
    """测试 1.10：设备分片"""
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    x = jnp.ones((len(devices) * 4, 32))
    if len(devices) == 1:
        x_sharded = jax.device_put(x, devices[0])
    else:
        x_sharded = jax.device_put(x, sharding.reshape(len(devices), 1))
    assert x_sharded.shape == (len(devices) * 4, 32)
    print("PASS: sharding")

if __name__ == "__main__":
    tests = [
        test_device_detection,
        test_jit_matmul,
        test_grad,
        test_vmap,
        test_random,
        test_device_put_get,
        test_dynamic_slice,
        test_conv,
        test_tree_operations,
        test_sharding,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
    print(f"\nResult: {passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
```

### 8.2 阶段二：Flax 网络层验证

**目标**：确认 Flax 神经网络层在 Intel GPU 上能正确初始化和前向传播。

**测试脚本** `test_02_flax_layers.py`：

```python
"""阶段二：Flax 神经网络层验证"""
import jax
import jax.numpy as jnp
import flax.linen as nn

def test_dense():
    """测试 2.1：Dense 层"""
    model = nn.Dense(64)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32)))
    output = model.apply(params, jnp.ones((4, 32)))
    assert output.shape == (4, 64)
    print("PASS: Dense layer")

def test_conv():
    """测试 2.2：Conv 层"""
    model = nn.Conv(features=32, kernel_size=(3, 3))
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 64, 64, 3)))
    output = model.apply(params, jnp.ones((4, 64, 64, 3)))
    assert output.shape == (4, 64, 64, 32)
    print("PASS: Conv layer")

def test_layer_norm():
    """测试 2.3：LayerNorm"""
    model = nn.LayerNorm()
    x = jnp.ones((4, 32))
    params = model.init(jax.random.PRNGKey(0), x)
    output = model.apply(params, x)
    assert output.shape == (4, 32)
    print("PASS: LayerNorm")

def test_group_norm():
    """测试 2.4：GroupNorm（ResNet 用）"""
    model = nn.GroupNorm(num_groups=4)
    x = jnp.ones((4, 8, 8, 32))
    params = model.init(jax.random.PRNGKey(0), x)
    output = model.apply(params, x)
    assert output.shape == (4, 8, 8, 32)
    print("PASS: GroupNorm")

def test_mlp_with_grad():
    """测试 2.5：MLP 前向 + 反向传播"""
    class SimpleMLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(256)(x)
            x = nn.tanh(x)
            x = nn.LayerNorm()(x)
            x = nn.Dense(256)(x)
            x = nn.tanh(x)
            x = nn.Dense(1)(x)
            return x

    model = SimpleMLP()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32)))

    def loss_fn(params, x):
        return jnp.mean(model.apply(params, x) ** 2)

    grad_fn = jax.jit(jax.grad(loss_fn))
    x = jax.random.normal(jax.random.PRNGKey(1), (16, 32))
    grads = grad_fn(params, x)
    grad_leaves = jax.tree_util.tree_leaves(grads)
    assert all(g.shape == p.shape for g, p in
               zip(grad_leaves, jax.tree_util.tree_leaves(params)))
    print("PASS: MLP forward + backward")

def test_resnet_block():
    """测试 2.6：ResNet 残差块"""
    class ResBlock(nn.Module):
        features: int
        @nn.compact
        def __call__(self, x):
            residual = x
            x = nn.Conv(self.features, (3, 3), padding="SAME")(x)
            x = nn.GroupNorm(num_groups=4)(x)
            x = nn.relu(x)
            x = nn.Conv(self.features, (3, 3), padding="SAME")(x)
            x = nn.GroupNorm(num_groups=4)(x)
            return nn.relu(x + residual)

    model = ResBlock(features=32)
    x = jnp.ones((2, 16, 16, 32))
    params = model.init(jax.random.PRNGKey(0), x)
    output = model.apply(params, x)
    assert output.shape == (2, 16, 16, 32)
    print("PASS: ResNet block")

def test_ensemble_vmap():
    """测试 2.7：Ensemble 网络（vmap）"""
    class SingleCritic(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(256)(x)
            x = nn.tanh(x)
            return nn.Dense(1)(x)

    VmapCritic = nn.vmap(
        SingleCritic,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=0,
        axis_size=2,
    )
    model = VmapCritic()
    x = jnp.ones((1, 32))
    params = model.init(jax.random.PRNGKey(0), x)
    output = model.apply(params, jnp.ones((16, 32)))
    assert output.shape == (2, 16, 1)  # (ensemble_size, batch_size, 1)
    print("PASS: Ensemble critic (vmap)")

if __name__ == "__main__":
    tests = [test_dense, test_conv, test_layer_norm, test_group_norm,
             test_mlp_with_grad, test_resnet_block, test_ensemble_vmap]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
    print(f"\nResult: {passed}/{len(tests)} tests passed")
```

### 8.3 阶段三：Distrax + Optax 验证

**测试脚本** `test_03_distrax_optax.py`：

```python
"""阶段三：概率分布与优化器验证"""
import jax
import jax.numpy as jnp
import distrax
import optax
import flax.linen as nn

def test_normal_distribution():
    """测试 3.1：正态分布采样与对数概率"""
    mean = jnp.zeros((4, 7))
    std = jnp.ones((4, 7))
    dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
    key = jax.random.PRNGKey(0)
    samples, log_probs = dist.sample_and_log_prob(seed=key)
    assert samples.shape == (4, 7)
    assert log_probs.shape == (4,)
    print("PASS: Normal distribution")

def test_tanh_transformed():
    """测试 3.2：Tanh 变换分布（SAC 策略核心）"""
    mean = jnp.zeros((4, 7))
    std = jnp.ones((4, 7)) * 0.5
    base_dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
    dist = distrax.Transformed(base_dist, distrax.Block(distrax.Tanh(), ndims=1))
    key = jax.random.PRNGKey(0)
    samples, log_probs = dist.sample_and_log_prob(seed=key)
    assert samples.shape == (4, 7)
    assert jnp.all(samples >= -1) and jnp.all(samples <= 1)
    print("PASS: Tanh transformed distribution")

def test_optax_adam():
    """测试 3.3：Adam 优化器更新"""
    model = nn.Dense(64)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32)))

    tx = optax.adam(learning_rate=3e-4)
    opt_state = tx.init(params)

    def loss_fn(params):
        return jnp.mean(model.apply(params, jnp.ones((8, 32))) ** 2)

    for step in range(5):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    final_loss = loss_fn(params)
    assert final_loss < loss  # Loss should decrease
    print("PASS: Adam optimizer (5 steps)")

def test_jit_training_step():
    """测试 3.4：JIT 编译完整训练步"""
    model = nn.Dense(7)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32)))
    tx = optax.adam(3e-4)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, batch, rng):
        def loss_fn(params):
            pred = model.apply(params, batch["obs"])
            mean = pred
            std = jnp.ones_like(mean) * 0.5
            dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
            log_prob = dist.log_prob(batch["actions"])
            return -jnp.mean(log_prob)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    batch = {
        "obs": jax.random.normal(jax.random.PRNGKey(1), (16, 32)),
        "actions": jax.random.normal(jax.random.PRNGKey(2), (16, 7)),
    }

    for i in range(3):
        params, opt_state, loss = train_step(
            params, opt_state, batch, jax.random.PRNGKey(i))

    print(f"PASS: JIT training step (final loss: {float(loss):.4f})")

if __name__ == "__main__":
    tests = [test_normal_distribution, test_tanh_transformed,
             test_optax_adam, test_jit_training_step]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
    print(f"\nResult: {passed}/{len(tests)} tests passed")
```

### 8.4 阶段四：项目级 Agent 单元测试

**目标**：不连接机器人，在 Intel 设备上独立验证 Agent 的创建和训练。

**测试脚本** `test_04_agent_standalone.py`：

```python
"""阶段四：Agent 独立创建与训练验证"""
import sys
sys.path.insert(0, "/path/to/hil-serl/examples")  # 调整为实际路径
sys.path.insert(0, "/path/to/hil-serl/serl_launcher")

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict

from serl_launcher.utils.launcher import make_sac_pixel_agent
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.utils.train_utils import concat_batches

def make_fake_obs_space_sample():
    """构造假的观测空间样本（模拟 FrankaEnv 输出）"""
    return {
        "state": np.random.randn(1, 17).astype(np.float32),
        "wrist_1": np.random.randint(0, 255, (1, 128, 128, 3), dtype=np.uint8),
        "wrist_2": np.random.randint(0, 255, (1, 128, 128, 3), dtype=np.uint8),
    }

def make_fake_batch(batch_size=32):
    """构造假的训练 batch"""
    batch = {
        "observations": {
            "state": np.random.randn(batch_size, 1, 17).astype(np.float32),
            "wrist_1": np.random.randint(0, 255, (batch_size, 2, 128, 128, 3), dtype=np.uint8),
            "wrist_2": np.random.randint(0, 255, (batch_size, 2, 128, 128, 3), dtype=np.uint8),
        },
        "next_observations": {
            "state": np.random.randn(batch_size, 1, 17).astype(np.float32),
        },
        "actions": np.random.randn(batch_size, 7).astype(np.float32).clip(-1, 1),
        "rewards": np.random.choice([0.0, 1.0], size=(batch_size,)).astype(np.float32),
        "masks": np.ones((batch_size,), dtype=np.float32),
        "dones": np.zeros((batch_size,), dtype=bool),
    }
    return frozen_dict.freeze(batch)

def test_agent_creation():
    """测试 4.1：Agent 创建（含 ResNet-10 权重加载）"""
    sample_obs = make_fake_obs_space_sample()
    sample_action = np.random.randn(7).astype(np.float32)

    agent = make_sac_pixel_agent(
        seed=42,
        sample_obs=sample_obs,
        sample_action=sample_action,
        image_keys=["wrist_1", "wrist_2"],
        encoder_type="resnet-pretrained",
        discount=0.97,
    )
    print(f"Agent created. Param count: {sum(x.size for x in jax.tree_util.tree_leaves(agent.state.params))}")
    print("PASS: Agent creation")
    return agent

def test_data_augmentation():
    """测试 4.2：数据增强"""
    img = jnp.ones((16, 2, 128, 128, 3))
    rng = jax.random.PRNGKey(0)
    result = batched_random_crop(img, rng, padding=4, num_batch_dims=2)
    assert result.shape == (16, 2, 128, 128, 3)
    print("PASS: Data augmentation (batched_random_crop)")

def test_agent_update(agent):
    """测试 4.3：单步训练更新"""
    batch = make_fake_batch(32)
    demo_batch = make_fake_batch(32)
    combined = concat_batches(batch, demo_batch, axis=0)

    networks_to_update = frozenset({"critic", "actor", "temperature"})
    new_agent, info = agent.update(combined, networks_to_update=networks_to_update)

    assert "critic_loss" in info
    assert "actor_loss" in info
    assert "temperature_loss" in info
    print(f"PASS: Agent update (critic_loss={float(info['critic_loss']):.4f}, "
          f"actor_loss={float(info['actor_loss']):.4f})")
    return new_agent

def test_sample_actions(agent):
    """测试 4.4：策略推理"""
    obs = jax.device_put(make_fake_obs_space_sample())
    rng = jax.random.PRNGKey(0)
    actions = agent.sample_actions(observations=obs, seed=rng, argmax=False)
    actions_np = np.asarray(jax.device_get(actions))
    assert actions_np.shape == (7,)
    print(f"PASS: sample_actions (output: {actions_np[:3]}...)")

def test_multiple_updates(agent):
    """测试 4.5：连续 10 步训练（模拟训练循环）"""
    critic_only = frozenset({"critic"})
    all_networks = frozenset({"critic", "actor", "temperature"})

    for step in range(10):
        batch = make_fake_batch(32)
        demo_batch = make_fake_batch(32)
        combined = concat_batches(batch, demo_batch, axis=0)

        if step < 9:  # N-1 steps: critic only
            agent, info = agent.update(combined, networks_to_update=critic_only)
        else:  # Last step: all networks
            agent, info = agent.update(combined, networks_to_update=all_networks)

    print(f"PASS: 10-step training loop (final critic_loss={float(info['critic_loss']):.4f})")

if __name__ == "__main__":
    print("=" * 60)
    print("Agent Standalone Test on Intel Device")
    print(f"Devices: {jax.devices()}")
    print("=" * 60)

    agent = test_agent_creation()
    test_data_augmentation()
    agent = test_agent_update(agent)
    test_sample_actions(agent)
    test_multiple_updates(agent)
    print("\nAll agent tests passed!")
```

### 8.5 阶段五：通信集成测试

**目标**：验证 Intel Learner 与 NVIDIA Actor 之间的跨机通信。

**测试步骤**：

```
步骤 1：在 Intel 设备上启动 Learner（仅启动 TrainerServer，不开始训练）
步骤 2：在 NVIDIA 机器上启动 Actor 连接测试
步骤 3：验证数据传输和参数广播
```

**Intel 设备上** `test_05a_learner_server.py`：

```python
"""阶段五A：Learner 端 TrainerServer 启动测试"""
import sys
sys.path.insert(0, "/path/to/hil-serl/examples")
sys.path.insert(0, "/path/to/hil-serl/serl_launcher")

import jax
import time
import numpy as np
from flax.core import frozen_dict

from serl_launcher.utils.launcher import make_sac_pixel_agent, make_trainer_config
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from agentlace.trainer import TrainerServer

import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace

# 构造观测空间
obs_space = DictSpace({
    "state": Box(low=-np.inf, high=np.inf, shape=(1, 17), dtype=np.float32),
    "wrist_1": Box(low=0, high=255, shape=(1, 128, 128, 3), dtype=np.uint8),
    "wrist_2": Box(low=0, high=255, shape=(1, 128, 128, 3), dtype=np.uint8),
})
act_space = Box(low=-1, high=1, shape=(7,), dtype=np.float32)

# 创建 replay buffer
replay_buffer = MemoryEfficientReplayBufferDataStore(
    obs_space, act_space, capacity=10000,
    image_keys=["wrist_1", "wrist_2"],
)

# 创建 agent
agent = make_sac_pixel_agent(
    seed=42,
    sample_obs=obs_space.sample(),
    sample_action=act_space.sample(),
    image_keys=["wrist_1", "wrist_2"],
)

# 统计回调
def stats_callback(type, payload):
    print(f"Received stats: {type} -> {list(payload.keys())}")
    return {}

# 启动 server
config = make_trainer_config()  # port 5588, broadcast 5589
server = TrainerServer(config, request_callback=stats_callback)
server.register_data_store("actor_env", replay_buffer)
server.register_data_store("actor_env_intvn", replay_buffer)  # 简化测试
server.start(threaded=True)

print(f"TrainerServer started on ports 5588/5589")
print(f"Waiting for Actor connection...")

# 等待数据到达
while len(replay_buffer) < 10:
    time.sleep(1)
    print(f"Buffer size: {len(replay_buffer)}")

# 广播参数
print("Publishing initial network parameters...")
server.publish_network(agent.state.params)
print("PASS: Server started and network published")

# 保持运行
try:
    while True:
        time.sleep(5)
        print(f"Buffer: {len(replay_buffer)} | Publishing params...")
        server.publish_network(agent.state.params)
except KeyboardInterrupt:
    server.stop()
    print("Server stopped")
```

**NVIDIA 机器上** `test_05b_actor_client.py`：

```python
"""阶段五B：Actor 端 TrainerClient 连接测试（在 NVIDIA 机器上运行）"""
import sys
sys.path.insert(0, "/path/to/hil-serl/examples")
sys.path.insert(0, "/path/to/hil-serl/serl_launcher")

import numpy as np
import time

from serl_launcher.utils.launcher import make_trainer_config
from agentlace.trainer import TrainerClient
from agentlace.data.data_store import QueuedDataStore

INTEL_IP = "192.168.x.x"  # ← 替换为 Intel 设备实际 IP

# 创建数据存储
data_store = QueuedDataStore(50000)
intvn_store = QueuedDataStore(50000)

# 创建客户端
config = make_trainer_config()
client = TrainerClient(
    "actor_env",
    INTEL_IP,
    config,
    data_stores={"actor_env": data_store, "actor_env_intvn": intvn_store},
    wait_for_server=True,
    timeout_ms=5000,
)
print(f"Connected to Learner at {INTEL_IP}")

# 注册参数接收回调
received_params = [False]
def on_params(params):
    received_params[0] = True
    param_count = sum(x.size for x in __import__("jax").tree_util.tree_leaves(params))
    print(f"Received network params ({param_count} parameters)")

client.recv_network_callback(on_params)

# 发送模拟 transitions
for i in range(20):
    transition = {
        "observations": {
            "state": np.random.randn(1, 17).astype(np.float32),
            "wrist_1": np.random.randint(0, 255, (1, 128, 128, 3), dtype=np.uint8),
            "wrist_2": np.random.randint(0, 255, (1, 128, 128, 3), dtype=np.uint8),
        },
        "actions": np.random.randn(7).astype(np.float32),
        "next_observations": {
            "state": np.random.randn(1, 17).astype(np.float32),
            "wrist_1": np.random.randint(0, 255, (1, 128, 128, 3), dtype=np.uint8),
            "wrist_2": np.random.randint(0, 255, (1, 128, 128, 3), dtype=np.uint8),
        },
        "rewards": float(np.random.choice([0.0, 1.0])),
        "masks": 1.0,
        "dones": False,
    }
    data_store.insert(transition)

# 同步到 server
success = client.update()
print(f"Data sync: {'SUCCESS' if success else 'FAILED'}")

# 发送统计信息
client.request("send-stats", {
    "environment": {"episode": {"r": 1.0, "l": 50, "t": 5.0}},
    "timer": {"sample_actions": 0.01, "step_env": 0.05, "total": 0.1},
})
print("Stats sent")

# 等待接收参数
time.sleep(10)
assert received_params[0], "Did not receive network params from Learner!"
print("\nPASS: Cross-machine communication test")
client.stop()
```

### 8.6 阶段六：端到端集成测试

**目标**：完整运行 `train_rlpd.py` 的 Learner 进程在 Intel 设备上、Actor 进程在 NVIDIA 机器上，验证实际训练流程。

**Intel 设备上**：
```bash
cd /path/to/hil-serl/examples/experiments/ram_insertion

# 设置 Intel GPU 环境
export JAX_PLATFORMS=xpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python ../../train_rlpd.py \
    --exp_name=ram_insertion \
    --learner \
    --checkpoint_path=intel_test_run \
    --demo_path=demo_data/xxx.pkl \
    --debug  # 禁用 WandB，简化测试
```

**NVIDIA 机器上**：
```bash
cd /path/to/hil-serl/examples/experiments/ram_insertion

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1

python ../../train_rlpd.py \
    --exp_name=ram_insertion \
    --actor \
    --checkpoint_path=intel_test_run \
    --ip=<INTEL_DEVICE_IP>
```

**验收标准**：

| 检查项 | 判定标准 |
|--------|---------|
| Learner 启动 | 打印 "starting learner loop"，无报错 |
| Actor 连接 | 打印 "starting actor loop"，成功连接 Learner |
| 数据传输 | Learner 端 replay_buffer 持续增长 |
| 训练循环 | Learner 端 tqdm 进度条正常推进 |
| 参数广播 | Actor 端策略行为随训练变化（非随机动作） |
| Critic Loss | 持续下降且不发散（不为 NaN/Inf） |
| 数值一致性 | 与 NVIDIA-only 训练相比，1000 步内 Q-value 量级相同（±50%） |
| Checkpoint | 可正常保存和加载 |

### 8.7 性能基准测试

在阶段四或阶段六通过后，运行以下基准测试：

```python
"""性能基准测试"""
import jax
import time

# 在 test_04 的基础上，度量训练速度
agent = test_agent_creation()

# 预热 JIT
batch = make_fake_batch(128)
agent, _ = agent.update(batch, networks_to_update=frozenset({"critic", "actor", "temperature"}))
jax.block_until_ready(agent)

# 基准测试
times = []
for i in range(100):
    batch = make_fake_batch(128)
    t0 = time.time()
    agent, info = agent.update(batch, networks_to_update=frozenset({"critic", "actor", "temperature"}))
    jax.block_until_ready(agent)
    times.append(time.time() - t0)

print(f"Training step: {np.mean(times)*1000:.1f}ms ± {np.std(times)*1000:.1f}ms")
print(f"Steps per second: {1/np.mean(times):.1f}")
# 参考值（NVIDIA RTX 3090）: ~15-30ms/step, ~30-65 steps/sec
```

---

## 9. 附录：关键数据结构定义

### 9.1 JaxRLTrainState 结构

```python
# 源码：serl_launcher/common/common.py:81-115
class JaxRLTrainState(flax.struct.PyTreeNode):
    step: int                     # 当前训练步
    apply_fn: Callable            # 模型前向函数（非 PyTree 节点）
    params: Dict                  # 当前网络参数
    target_params: Dict           # 目标网络参数
    txs: Dict[str, optax.GradientTransformation]  # 优化器（非 PyTree 节点）
    opt_states: Dict[str, Any]    # 优化器状态
    rng: jax.Array                # 内部 PRNG 状态
    epsilon: float = 0.0          # epsilon-greedy 参数
```

### 9.2 SACAgent 结构

```python
# 源码：serl_launcher/agents/continuous/sac.py:21-31
class SACAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict  # 非 PyTree 节点，包含超参数
```

### 9.3 训练超参数（RAM Insertion 任务）

```python
batch_size = 256          # 总 batch（128 online + 128 demo）
discount = 0.97           # 折扣因子
cta_ratio = 2             # UTD ratio（2次 critic 更新 + 1次全更新）
steps_per_update = 50     # 每 50 步广播参数
training_starts = 100     # 至少 100 条数据才开始训练
replay_buffer_capacity = 200000
max_steps = 1000000
encoder_type = "resnet-pretrained"   # 冻结 ResNet-10 + 可训练 pooling
actor_lr = 3e-4           # Actor 学习率
critic_lr = 3e-4          # Critic 学习率
temperature_init = 0.01   # SAC 温度初始值
soft_target_update_rate = 0.005  # Polyak τ
critic_ensemble_size = 2  # 两个 Critic 网络
```

### 9.4 网络规模参考

| 网络 | 参数量（估算） | 说明 |
|------|--------------|------|
| ResNet-10 Encoder (×2 cameras) | ~5M × 2 = ~10M | 冻结，不需要梯度 |
| Spatial Embeddings (×2) | ~200K × 2 = ~400K | 可训练 |
| Actor MLP (256, 256) | ~200K | 可训练 |
| Critic MLP (256, 256) × 2 ensemble | ~200K × 2 = ~400K | 可训练 |
| Temperature | 1 | 标量参数 |
| **总参数** | **~11M** | 其中 ~10M 冻结 |
| **可训练参数** | **~1M** | 梯度计算范围 |

---

## 文档结束

如有技术问题，请联系项目维护方。
