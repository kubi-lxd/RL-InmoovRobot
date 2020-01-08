# rl_baselines.train.py
强化学习训练代码笔记
## main

### 输入配置
典型例子：


* --algo  算法选择 选项在rl_baselines.registry.py中的registered_rl字典中定义
其下方检测每个算法model是否是StableBaselinesRLObject的子类 所以如果自己添加算法包 要参看rl_algorithm的格式
* --env 环境选择 选项在environments.registry.py中的registered_env字典中定义
environments.registry.py 中会将本项目中的文件在gym.envs中注册
* --seed 种子
* --episode-window 影响rl_baselines.visualize.episodePlot函数 应该是个平滑窗
* --log-dir 日志路径 存储智能体的日志和模型
* --num-timesteps 好像没有使用
* --srl-model srl表示模型选择 state_representation中的registry中的字典定义
* --num-stack 好像没有使用
* --action-repeat  输入env_kwargs
* --port  vistom的观察端口
* --no-vis  关闭vistom功能  action 可以不输后面的指令激活action  --no-vis 与--no-vis = True等同
* --shape-reward  暂时不懂
* -c  是否连续运动
* -joints  直接控制关节 而不是末端逆解
* -r  是否随机目标  进入env_kwargs
* --srl-config-file  srl配置文件路径
* --hyperparam  超参 引入algo.train
* --min-episodes-save  判断出现最好模型的最小存储回合  平均多少个模型都超过之前了 就保存一下
* --latest   读取最后一个model
* --load-rl-model-path  模型读取路径配置
* -cc  以下若干种c是一种连续模型的方法  被赋予了env_kwargs
* -sqc
* -ec
* -chc
* -esc
* --teacher-data-folder  policy distillation 检测  判断示教策略是否有效
* --epochs-distillation   暂时未使用
* --distillation-training-set-size  暂时未使用
* --perform-cross-evaluation-cc     交叉验证
* --eval-episode-window   影响 EPISODE_WINDOW_DISTILLATION_WIN
* --new-lr  作为超参数输入algo.train

### 执行逻辑

1. 首先是args把上述用户输入配置拿下来，并做一定的规则检查

2. 读取srl_model.yaml 获取srl模型配置信息
3. 调用configureEnvAndLogFolder完成srl路径配置
4. ENV_NAME  ALGO_NAME等关键全局变量配置
5. 提取相关信息构建env_kwargs
6. 记录存储args
7. 构建超参数
8. 开始训练 调用callback
9. callback完成模型储存

### env_kwargs

* is_discrete
* action_repeat
* random_target
* muiti_view
* shape_reward
* action_joints
* srl_model
* use_srl
* srl_model_path
* simple_continual_target,circular_continual_move,square_continual_move,eight_continual_move,chasing_continual_move,escape_continual_move

## rl_baselines.rl_algorithm.ppo2.train

这里是上面的train文件直接调用的训练函数

进来以后填充param_kwargs 接着args和env_kwargs 以及超参传递到父类执行

其父类StableBaselinesRLObject内的train函数为直接继承函数

## rl_baselines.baseclasses.StableBaselinesRLObject.train

### env

env_kwargs经过makeEnv函数再经过rl_baselines.utils.createEnvs处理得到envs gym环境

其中createEnvs调用了environments.utils内的makeEnv函数

environments.utils.makeEnv函数复写了_make函数实现了环境的注册和创建（TODO：gym相关）

env再经过DummyVecEnv等包装（暂时不懂） 出场

### model

从model_class生成self.model

调用model.learn开始训练

由于PPO2Model自身初始化函数有

```python
super(PPO2Model, self).__init__(name="ppo2", model_class=PPO2)
```

model_class为stable_baselines.PPO2  拿到了模型

```python
super(PPO2, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,                           _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs)
```

这里继承了ActorCriticRLModel的init  ActorCriticRLModel里的init

```python
super(ActorCriticRLModel, self).__init__(policy, env, verbose=verbose, requires_vec_env=requires_vec_env,                                         policy_base=policy_base, policy_kwargs=policy_kwargs)
```

又继承BaseRLModel的init  多重继承下    相关信息都被配置给了model

### learn

这里的learn 实际上是多重继承的PPO2model的learn

```python
def learn(self, total_timesteps, callback=None, seed=None, log_interval=1, tb_log_name="PPO2",          reset_num_timesteps=True):
```

发生位置在stable_baselines.ppo2.ppo2.PPO2.learn

内部进行参数配置和种子配置

调用Runner拿到env环境

#### Runner

这里重要！

```python
self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)
```

实际是这里与环境交互的 为了拿到必要的数据  可以对比kuka_env来看

（待补充 跳转到env_note.md）



按batch开始循环 mb_loss_vals是局部损失

mb_loss_vals通过_train_step得到

### _train_step

内部简单粗暴 sess run了一下 该有的都有了

那么图是在那建的？转到setup_model

### PPO2.init

这个初始化函数中声明了self.graph 以及self.sess

当_init_setup_model为真时，调用setup_model函数

### setup_model

这里调用了self.policy 它是ActorCriticPolicy的子类

涉及到DiagGaussianProbabilityDistributionType

有专属的policy数据流（待补充）

### 模型再训练

**例子**
首先我们训练一个模型，他将会记录在目录`logs/test_finetune/**EnvName**` 之中

```python -m rl_baselines.train --env InmoovGymEnv-v0  --srl-model ground_truth  --log-dir logs/test_finetune/```

如果我们希望对这个模型重新训练，那么可以使用以下语句

```python -m rl_baselines.train --env InmoovGymEnv-v0  --srl-model ground_truth  --log-dir logs/test_finetune/ --load-rl-model-path logs/test_finetune/**EnvName**/ground_truth/ppo2/**logtime**/ppo2.pkl```

需要注意的是，一定要使用`ppo2.pkl`，而不是`ppo2_model.pkl`， model文件里面事实上只是存了他的几个超参数，前者才是真正的模型，当你到文件目录下看的时候也可以通过文件大小判断

另外需要注意的是，再训练的learning rate会默认调到原来的百分之一，事实上的RL训练一般默认线性的lr decay，如果不记得当初训练断点在哪里，直接重新接上去训练效果肯定会与未发生间断训练的结果有些许出入
当然也可以通过参数`--new-lr`来修改，但是目前的版本我不能保证正确性，如果有必要后期会进行进一步维护

