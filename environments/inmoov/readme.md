# 看什么呢，还没见过个readme？

---------------------------------------------
A simple introduction of the code structre
---------------------------------------------------
`inmoov.py`包含了机器人的一些属性特征，其中里面有control的功能，也包含一个debugger模式

`immoov_p2p.py`是一个包含番茄和机器人的环境，他继承了SRLGym的结构，使用了`inmoov.py`中的机器人定义和控制功能

`joints_registry`包含了可以控制的joints信息


To run some test code for the inmoov environement
----------------------------------------------------
- 请先退出到根目录，也就是`RL-InmoovRobot`目录
- 执行 ``python -m environments.inmoov.test_inmoov``
- 如果要debug `inmoov.py`，请使用test_inmoov函数， 如果是debug `immoov_p2p.py` 文件，请使用test_inmoov_gym函数。

对于inmoov debug模式下不能显示全部joints的情况，暂时还无法解决，我给pybullet发了一个[issue](https://github.com/bulletphysics/bullet3/issues/2519)。目前比较好的解决方案是在
`joints_registry`里面改一下，注释掉不想看到的joints。

To train your first model (bugs alert!)
--------------------------------
环境还没有完全干净，里面可能存在些许问题，但是目前的进度允许我们进行简单的训练。
同样，请先到达项目的根目录。

- launch a ``visdom`` server to monitor the training process by: 
```
python -m visdom.server
```

- launch the training process:
```
python -m rl_baselines.train --env InmoovGymEnv-v0 --srl-model ground_truth --algo ppo2 --log-dir logs/ --num-timesteps 2000000
```


Remote control
----------------
现在的inmoov环境可以对服务器进行远端控制了。

#### 配置环境：
- 服务器要求完整的python环境，可以参考robotics-srl的环境配置；
- 客户端，即自己电脑这一端，需要安装的主要有：
```
zmq #用于实现网络交互
matplotlib # 用于画图，呈现双目视角
cv2 # 用于实现简易的滚动条ui设计
numpy #数字计算及array处理
```

#### 运行代码：
先修改user_config.json内部的配置数据，用户名等信息

在服务器端运行(由于当初命名问题，服务器端命名为client，后续可能会修改)
```shell script
python -m environments.inmoov.inmoov_client
```
在客户端（个人PC），同样在项目根目录，需要执行代码：
```shell script
python -m environments.inmoov.inmoov_server
```
虽然是zmq的PAIR传输，理论上不会出现丢包的问题，但是依旧建议先运行服务器端（信息的主要发送端）。

#### 执行效果
期待在自己的PC上（客户端）出现一个opencv构成的拖动条，一个matplotlib窗口提供双目视觉信息。

中间件输入输出接口：
----------------
可以从`inmoov_server.py`中寻找，当然前提是开启client作为数据发送端，才能够形成数据交互，这个途径的好处是可以直接在windows上工作，免去了安装环境的痛苦，
困难点在于，网速不好的时候，交互过程和数据发送非常缓慢。
其中变量：`joint_state, left_px, right_px, reward, done, effector_position`均是预留的数据来源

也可以从`inmoov_client.py`中寻找，在主程序main里面也可以看到，data变量是由`robot.server_step(msg[command])`得来的，其中
函数通过server_step来实现环境交互的功能。

