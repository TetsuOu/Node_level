

## 更新日志
### 3.1
经统计，共989366个原子，其中922176个原子不发生变化，8736个原子为atom类型反应中心，58454个原子为bond类型反应中心。
考虑在CrossEntropyLoss中加入权重
```
weight = torch.tensor([1., 100.]) 
#如此设置，跑了50多个epoch，eval acc仍在37%上下
```

### 2.28
更改标签，改为二分类，eval acc达40%，train acc为84%左右

### 2.27
修改损失函数，eval acc仍在38%左右，train acc为84%左右

