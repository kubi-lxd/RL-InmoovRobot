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
 
对于有一个障碍物的体系，通过shaped且dense reward训练似乎成功

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
