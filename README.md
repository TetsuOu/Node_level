

## 更新日志
### 3.3
尝试了Focal loss，以期解决样本不均衡问题
```
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='sum'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict)
        loss = -self.alpha*(1-pt)**self.gamma*target*torch.log(pt)-(1-self.alpha)*pt**self.gamma*(1-target)*torch.log(1-pt)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

loss_func = BCEFocalLoss()
loss = 0
for i in range(len(tup)):
    loss += loss_func(tup[i], F.one_hot(label[i].long()))
    loss /= len(tup)
```
尝试了若干组参数
但是效果并不是很好，顺带着把train acc也降的很低了。。
可能是Focal loss用的不对

### 3.2
如下设置权重与损失函数
```python
weight = torch.tensor([922176, 8736+58454], dtype=torch.float32)
weight = [sum(weight)/x for x in weight]
weight = [x/sum(weight) for x in weight]
loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight)).to(device)

loss = 0

for i in range(len(tup)):
    for j in range(len(tup[i])):
        loss += loss_func(tup[i][j], label[i][j].long())

loss /= len(label)
```
200个epoch，train acc为67.8%，eval acc为40%

### 3.1
经统计，共989366个原子，其中922176个原子不发生变化，8736个原子为atom类型反应中心，58454个原子为bond类型反应中心。
考虑在CrossEntropyLoss中加入权重
```python
weight = torch.tensor([1., 100.]) 
#如此设置，跑了50多个epoch，eval acc仍在37%上下
```

### 2.28
更改标签，改为二分类，eval acc达40%，train acc为84%左右

### 2.27
修改损失函数，eval acc仍在38%左右，train acc为84%左右

