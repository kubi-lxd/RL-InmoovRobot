# RL-InmoovRobot
Humanoid robot Inmoov performs complex tasks in virtual environments using Reinforcement Learning(RL) and S-RL Toolbox

## 2019.12.15

- 实现了点到点的训练,测试的话使用:
```
sh run_train.sh
```
- enjoy训练结果的话使用:
```
sh run_trainresult.sh
```
- 修正部分冲突后合并,可能目前master pull下来会带冲突,修改的部分:
1. inmoov.py内的urdf被修改为了inmoov_colmass.urdf 这是一个删除了mass和修改了bicep运动范围的文件,为了避免出错保留了原urdf文件
2. inmoov.py添加了reset_joints方法
3. inmoov_p2p.py修改了reset部分,通过重置关节位置替代重置环境,测试通过,提速明显
4. 修改了gitignore文件 删除了对urdf_robot文件夹的屏蔽
- 训练特性:
使用距离作为reward,dv被设置为1.2的情况下,6线程训练一般机器2-3小时左右收敛,收敛时的回报约为-75左右,经过40万个steps
