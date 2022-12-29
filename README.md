# README

结果和测试样例的输出保存在result文件夹中，对应模型名称的demo文件夹中

#### 若想测试X-net

```
$> python eval.py --config ./config/config.yml
```

#### 若想测试Dilation X-net
```
$> python eval.py --config ./config/config_4.yml
```

#### 若想从头训练
先将数据保存在"../BWE_dataset"文件夹中
```
$> rm -rf ./result
$> run.sh
```