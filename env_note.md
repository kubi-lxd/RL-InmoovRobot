# Kuka button Env 笔记

这个python文件内部将实现一个gymEnv文件，包含一定接口，以符合强化学习训练的要求

上接PPO2.py内的Runner.run 里面仅有两行与env发生了交互

为了运行gym环境 PPO算法会检测actions是否超出env.action_space范围 

KukaButtonGymEnv类里面有一段action_space的配置

离散时spaces.Discrete、连续时spaces.Box

np.clip函数将actions限制在space范围内  这里是行为空间就是每次动作不能太大 不是运动空间

 self.env.step(clipped_actions) 函数得到 self.obs[:], rewards, self.dones, infos

# 运行逻辑

train最终调用了ppo2.learn

ppo2.learn里调用了Runner.run

Runner实际通过Kuka button Env来获取rewards

这里通过env.step获得reward

因此进入env类的step函数

## 全局变量

* MAX_STEPS  一次回合的最大步骤数
* N_CONTACTS_BEFORE_TERMINATION  
* N_STEPS_OUTSIDE_SAFETY_SPHERE
* RENDER_HEIGHT
* RENDER_WIDTH
* Z_TABLE
* N_DISCRETE_ACTIONS
* BUTTON_LINK_IDX
* BUTTON_GLIDER_IDX
* DELTA_V    这个是离散模式时的速度设置
* DELTA_V_CONTINUOUS   这个是连续模式时的速度设置
* DELTA_THETA   这个是关节模式时的速度设置
* RELATIVE_POS  使用相对位置 是个布尔量
* NOISE_STD  噪声量 实际是高斯函数的方差部分 用来给行为加抖动 
* NOISE_STD_CONTINUOUS 这个是连续控制模式时的噪声量
* NOISE_STD_JOINTS 这个是关节控制时的噪声量
* N_RANDOM_ACTIONS_AT_INIT
* BUTTON_DISTANCE_HEIGHT
* CONNECTED_TO_SIMULATOR

暂不明含义的留空

**getGlobals** 函数将一次性获取上面的全局变量（一个大字典）

## KukaButtonGymEnv

这个是本文件构建的主体环境类 继承SRLGymEnv



### init

init的初始化参数列表及含义

```python
:param urdf_root: (str) Path to pybullet urdf files
:param renders: (bool) Whether to display the GUI or not
:param is_discrete: (bool) Whether to use discrete or continuous actions
:param multi_view :(bool) if TRUE -> returns stacked images of the scene on 6 channels (two cameras)
:param name: (str) name of the folder where recorded data will be stored
:param max_distance: (float) Max distance between end effector and the button (for negative reward)
:param action_repeat: (int) Number of timesteps an action is repeated (here it is equivalent to frameskip)
:param shape_reward: (bool) Set to true, reward = -distance_to_goal
:param action_joints: (bool) Set actions to apply to the joint space
:param record_data: (bool) Set to true, record frames with the rewards.
:param random_target: (bool) Set the button position to a random position on the table
:param force_down: (bool) Set Down as the only vertical action allowed
:param state_dim: (int) When learning states
:param learn_states: (bool)
:param verbose: (bool) Whether to print some debug info
:param save_path: (str) location where the saved data should go
:param env_rank: (int) the number ID of the environment
:param srl_pipe: (Queue, [Queue]) contains the input and output of the SRL model
:param srl_model: (str) The SRL_model used
```
 获得一些必要信息的函数

* getSRLState 用视频流计算状态值
* getTargetPos 获取目标坐标（这里是拿到按钮）
* getJointsDim获取关节维度数
* getGroundTruthDim 获得真实参数维度
* getGroundTruth  获得真实参数
* getArmPos 获得机器人末端位置

### reset

环境的初始化

重新读取地面urdf

读取桌子  随机放置按钮

读取按钮urdf 拿到位置 设置重力

调用kuka的初始化函数

清零计数器 机械臂运行500步去趋近初始化位置

随机初始化机械臂位置

存一下模型 然后就把观测值返回

（这个函数的调用位置暂不明 应该是有调用 待补充）

### step

```**重要函数**```

输入action

**离散条件下：**

三个方向dx dy dz  一次偏移DELTA_V action负责选择运动方向 得到real_action

**关节控制模式下：**

```python
self._kuka = kuka.Kuka  之前这里配置了机器人模型
```

拿到所有要动的关节 是个列表

d_theta受DELTA_THETA配置，是角速度，并受NOISE_STD_JOINTS加噪声

```python
real_action = list(action * d_theta + arm_joints) + [0, 0]
```

这里得到真实的行为 这里的action应该是个7自由度（等同kuka控制的自由度）的列表，并表示了方向等

**连续控制模式下：**

速度受DELTA_V_CONTINUOUS设置 NOISE_STD_CONTINUOUS加噪声

这里的action是一个3自由度的列表  用来计算dx dy dz

上面计算完real_action以后（这里都变成float了） 调用step2函数

### step2

```**重要函数**```

* 回报
* done 是否回合结束标记
* {}  空字典 应该是本来想写一些打印消息的

那么下一个重要的点就是这里的reward函数

### _reward

```**重要函数**```

reward逻辑

首先拿到机器人末端位置

计算和button的distance（空间距离）

使用getContactPoints判断机器人是否与按钮和桌面有接触点

如果只接触按钮 reward =1   n_contacts+1

如果没接触且distance超过范围 或接触桌面：

reward =-1  n_steps_outside+1

如果没接触 但比较近 且没碰桌面

reward =0

然后是终止逻辑

如果撞了桌面 或 n_contacts（接触按钮）超过一定步数 或 n_steps_outside（在远处游荡）超过步数  终止此回合

**一些特殊情况的回报：**

如果设置为_shape_reward

离散条件回报值是负距离

其他条件是如果终止时回报大于0 设置为50的回报

终止时回报小于0  设置为-250的回报

该特殊情况的回报可能较少使用

----

以上实现了回报值的计算



