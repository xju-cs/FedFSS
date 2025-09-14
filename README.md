# 联邦说话人识别

#### FedFSS 介绍

软件架构说明  

data目录下包含训练列表，使用时需替换数据路径  

fedavg为FedAvg方法  
moon为MOON方法  
fedprox为fedProx方法  
****_lama为添加权重再分配DWA方法  
****_gamma为添加收缩因子方法 

#### 运行
python train_fed_anchor_lama_gamma.py 

#### New
新增config.py文件，可统一设置参数

