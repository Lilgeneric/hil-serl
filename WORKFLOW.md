# HIL-SERL: RAM Insertion 复现指南

本文档旨在记录并复现 [HIL-SERL](https://github.com/rail-berkeley/hil-serl) (Human-in-the-Loop SERL) 的官方入门任务：**RAM Insertion (内存条插入)**。本教程涵盖环境安装、硬件配置、数据采集及强化学习训练的全流程。

## 📋 目录

1. [环境依赖与安装](https://www.google.com/search?q=%231-%E7%8E%AF%E5%A2%83%E4%BE%9D%E8%B5%96%E4%B8%8E%E5%AE%89%E8%A3%85)
2. [硬件准备与服务器启动](https://www.google.com/search?q=%232-%E7%A1%AC%E4%BB%B6%E5%87%86%E5%A4%87%E4%B8%8E%E6%9C%8D%E5%8A%A1%E5%99%A8%E5%90%AF%E5%8A%A8)
3. [实验参数配置](https://www.google.com/search?q=%233-%E5%AE%9E%E9%AA%8C%E5%8F%82%E6%95%B0%E9%85%8D%E7%BD%AE)
4. [训练流程](https://www.google.com/search?q=%234-%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B)
* [阶段一：训练奖励分类器](https://www.google.com/search?q=%23%E9%98%B6%E6%AE%B5%E4%B8%80%E8%AE%AD%E7%BB%83%E5%A5%96%E5%8A%B1%E5%88%86%E7%B1%BB%E5%99%A8-reward-classifier)
* [阶段二：录制人类演示](https://www.google.com/search?q=%23%E9%98%B6%E6%AE%B5%E4%BA%8C%E5%BD%95%E5%88%B6%E4%BA%BA%E7%B1%BB%E6%BC%94%E7%A4%BA-demonstrations)
* [阶段三：策略训练与人工干预](https://www.google.com/search?q=%23%E9%98%B6%E6%AE%B5%E4%B8%89%E7%AD%96%E7%95%A5%E8%AE%AD%E7%BB%83%E4%B8%8E%E4%BA%BA%E5%B7%A5%E5%B9%B2%E9%A2%84-policy-training)



---

## 1. 环境依赖与安装

在运行代码之前，需要安装必要的 Python 库并配置硬件访问权限。

### 1.1 安装 Python 依赖

```bash
# 安装 SpaceMouse 驱动库
pip3 install pyspacemouse

# 安装 RealSense 摄像头库
pip install pyrealsense2

```

### 1.2 配置 USB 权限 (Udev Rules) ⚠️ **至关重要**

仅仅安装 Python 包不足以让代码直接访问硬件，必须配置 Udev 规则，否则会报错（如 `RuntimeError: No device detected` 或权限拒绝）。

1. **下载规则文件**：
```bash
sudo curl -o /etc/udev/rules.d/99-realsense-libusb.rules https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules

```


2. **重载规则并生效**：
```bash
sudo udevadm control --reload-rules && udevadm trigger

```


> **提示**：执行完上述命令后，请**拔掉并重新插入**摄像头 USB 接口以确保规则生效。


3. **hilserl其他环境安装**：
* **核心仓库**: [HIL-SERL GitHub](https://github.com/rail-berkeley/hil-serl)
* **机器人基础设施**: [SERL Robot Infra](https://github.com/rail-berkeley/hil-serl/blob/main/serl_robot_infra/README.md)
* **Franka 控制器**: [SERL Franka Controllers](https://github.com/rail-berkeley/serl_franka_controllers)

---

## 2. 硬件准备与服务器启动

在运行任何 AI 代码前，必须启动控制机器人的 HTTP 服务器。

### 2.1 硬件检查

* **Franka 机械臂**：确保已解锁。
* **FCI**：在 Franka Desk 界面中激活 **Franka Control Interface (FCI)**。
* **网络**：确保工作站与机器人在同一局域网内。

### 2.2 启动 Robot Server

1. **修改启动脚本**：
编辑 `serl_robot_infra/robot_servers/launch_right_server.sh`。
* 修改 **IP 地址**（本机 IP 和 机器人 IP）。
* 修改 **catkin_ws** 路径。


2. **运行脚本**：
```bash
bash serl_robot_infra/robot_servers/launch_right_server.sh

```


3. **验证状态**：
终端应显示服务器启动成功。你可以尝试使用 `curl` 命令控制夹爪开合来验证连接。

---

## 3. 实验参数配置

这是最关键的步骤，需编辑配置文件：`examples/experiments/ram_insertion/config.py`。

| 参数项 | 说明与操作 |
| --- | --- |
| **SERVER_URL** | 填入步骤 2 启动的 Flask 服务器地址。<br>

<br>例如：`http://127.0.0.1:5000` 或局域网 IP。 |
| **REALSENSE_CAMERAS** | 填入两个腕部摄像头 (Wrist Cameras) 的序列号 (Serial Number)。<br>

<br>可通过 `RealSense Viewer` 软件查看。 |
| **关键位姿 (Poses)** | 需要定义 `TARGET_POSE` (插入完成), `GRASP_POSE` (抓取), `RESET_POSE` (复位)。 |

### 🔍 如何获取位姿坐标？

手动将机械臂拖动到目标位置，然后运行以下命令获取当前坐标（Euler）：

```bash
curl -X POST http://<FRANKA_SERVER_URL>:5000/getpos_euler

```

将返回的数据填入 `config.py` 对应的字段中。

---

## 4. 训练流程

### 阶段一：训练奖励分类器 (Reward Classifier)

机器人需要一个“判官”来判断当前动作是成功还是失败。

1. **采集成功/失败数据**：
```bash
cd examples
python record_success_fail.py --exp_name ram_insertion --successes_needed 200

```


* **操作方法**：按住空格键 = “成功”（状态）；松开空格键 = “失败”或“进行中”。
* **数据技巧**：不仅要采集正常的失败，还要采集“花式失败”（如对准了但没插进去、悬空等）。负样本数量建议是正样本的 2-3 倍。


2. **训练网络**：
```bash
cd experiments/ram_insertion
python ../../train_reward_classifier.py --exp_name ram_insertion

```



### 阶段二：录制人类演示 (Demonstrations)

RL 需要一些成功的示例作为“热启动”数据。

1. **运行录制脚本**：
```bash
python ../../record_demos.py --exp_name ram_insertion --successes_needed 20

```


2. **操作要求**：使用 SpaceMouse 控制机器人完成 **20 次** 完美的插入操作。
3. **自动重置**：如果阶段一的分类器训练良好，当你成功插入时，脚本会自动识别并 Reset 机器人。

### 阶段三：策略训练与人工干预 (Policy Training)

这是 HIL-SERL 的核心，需要同时开启两个终端运行 Actor 和 Learner。

1. **准备脚本**：
编辑 `run_actor.sh` 和 `run_learner.sh`。
* `checkpoint_path`: 设置模型保存路径。
* `demo_path`: 指向阶段二录制的演示数据路径。


2. **启动训练**：
* **终端 1 (Actor)**：负责与环境交互
```bash
bash run_actor.sh

```


* **终端 2 (Learner)**：负责更新策略网络
```bash
bash run_learner.sh

```




3. **Human-in-the-Loop 教学指南** 🎮
* **初期表现**：机器人动作会比较随机。
* **干预时机**：当机器人离目标太远、动作危险或长时间卡住时。
* **如何干预**：使用 SpaceMouse 接管，将机器人带到插槽附近或修正姿态，然后**松手**让其尝试自动插入。
* **教学原则**：不要全程代劳。只在它完全错误时纠正，随着熟练度提升，减少干预频率。
* **预期时间**：约 **1.5 小时** 可达到接近 100% 成功率。

---