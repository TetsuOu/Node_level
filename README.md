

## 更新日志
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

