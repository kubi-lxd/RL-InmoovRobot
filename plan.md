# 机器人环境构建计划

## 构造InmoovOneArmTomatoGymEnv类

构造参数仿照kuka_button （后面再缩减）

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
**功能实现：**

环境中应该是一个机器人urdf 和一颗番茄树

实现单臂点到点的强化学习

分为2种：

gruond_truth下 末端逆解的控制  多joint的控制

TODO：2种  raw_pixels下末端逆解的控制 多joint的控制（这个是回报不好定 待讨论）

**向Tete确认的问题：**

1. 机器人右臂接近果实的话  应该参与控制的joint序号列表
2. 机器人末端原代码里使用getLinkState函数确定末端坐标 是否有变化
3. 番茄树的果实位置能否用函数拿到 还是暂时定一个坐标
4. getContactPoints 判断是否触碰到果实对番茄树目标是否有效  如果无效可能考虑先做button了 只先替换一下机器人
5. 本应发生在inmoov.py里的joint属性我这边先定义 包括ll,ul,jr,rp,jd  这个应该在问题1之后
6. 环境reset的话准备先不做果树的随机化 果实就固定的 先机器人随机初始状态
7. reward的话参考button去掉桌子  即离的很远-1   较近没碰0   碰到1
8. 考虑到urdf自身带有避碰功能  这样inmoov.applyAction(错误的导致碰撞的action)之后 应该模型是不动的 是这个样子么 确认一小下  因为这样的话将来连接实机应该就是继承这个applyAction函数就好了  把控制命令发下去  这样需要判断避碰发生的环节  以及如何判断自己是不是已经碰了 可能有负回报
9. 这样的话需要进行单元检测 写好这一坨以后再确认怎么联调

包含类函数

### getSRLState getTargetPos getJointsDim getTargetPos getArmPos getGroundTruthDim getGroundTruth

这里的话确认问题1、2、3就可以写

### reset

这里是环境的复位函数  从原代码的写法看这种是支持实机动作复位的 准备参考

### getExtendedObservation

这里是取rgb通道 应该参看inmoov内的摄像头设置就可以写

### step step2

这里实现 gruond_truth 末端逆解的控制  多joint的控制

### 





