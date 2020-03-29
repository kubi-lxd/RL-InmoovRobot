训练方法记录
 -------

### 方案一

对于有一个障碍物的体系，通过sparse reward训练失败

``` python
contact_button = 10 * int(len(p.getContactPoints(self._jaka_id, self.button_uid)) > 0)
contact_obs = - len(p.getContactPoints(self._jaka_id, self.obstacle_id))
r = contact_button + contact_obs
```
无论是ground truth还是raw pixel都不能获得超过0的reward，注意这里的obstacle是硬的，固定的

### 方案二

对于有一个障碍物的体系，通过shaped且dense reward训练似乎成功,reward最终能够达到2500左右

``` python
contact_button = 10 * int(len(p.getContactPoints(self._jaka_id, self.button_uid)) > 0)
contact_obs = len(p.getContactPoints(self._jaka_id, self.obstacle_id))

r = 0
if contact_obs > 0:
    self.terminated = True
    r = - contact_obs
else:
    distance = np.linalg.norm(self.getGroundTruth() - self.getTargetPos())
    if distance < 0.1:
        r += 10
    else:
        r = 1 / distance
```


### 方案三
max_step=250, 对于简谐运动的障碍物，我们给出这样的reward function，旨在加速机器人碰到棍子的速度，走的步数越多，回报越少

```python
r = 0
if contact_obs > 0:
    self.terminated = True
    r = - 10 * contact_obs
else:
    distance = np.linalg.norm(self.getGroundTruth() - self.getTargetPos())
    if distance < 0.1:
        r += 10
        self.terminated = True
    else:
        r = 1 / distance - (self._step_counter / self.max_steps)
```

### 方案四
max_step=250, 对于某个区域随机运动障碍物，我们给出这样的reward function，旨在加速机器人碰到棍子的速度，走的步数越多，回报越少

```python
r = 0
if contact_obs > 0:
    self.terminated = True
    r = - 10 * contact_obs
else:
    distance = np.linalg.norm(self.getGroundTruth() - self.getTargetPos())
    if distance < 0.1:
        r += 10
        self.terminated = True
    else:
        r = 1 / distance - (self._step_counter / self.max_steps)
```
