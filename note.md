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

