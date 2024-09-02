# General Execute Architecture

![img/gea_logo.png](https://github.com/HackGiter/General-Execute-Architecture)

## 简介(Introduction)
作为一种通用的通用模型运行工具，包括训练、评估以及推理等任务，还在完善中，主要是我平时按需求添加功能。

## 特性(Features)
1. 主要是提供更加统一、自由度更高的规划模式，更快捷、方便地调整以及规划包括数据、训练、评估以及推理等任务。
2. 为用户提供更加统一、完备的实验环境，将包括数据处理、规划训练、训练中/后续评估、推理以及设计评价指标等内容整合进一个项目中。
3. 自定义内容都提供默认执行方法

### 训练(Training)
- execute_train_process:Callable(accelerator, model, optimizer, lr_scheduler, train_dataloader, eval_dataloaders, train_state, callback_handler, **kwargs): 规划训练过程
- compute_loss:Callable(model, batch, **kwargs): 规划 model & inputs 计算得到loss & metrics
- prepare_inputs_fn:Callable(inputs, **kwargs): 处理 model 即将接受的数据
- exec_eval_fn:Callable(model, eval_dataloader, train_state, description, **kwargs): 自定义训练中评估过程

### 模型和数据(Models & Datas)
- get_model_fn:Callable(model_args, **kwargs): 定义模型和tokenizer
- config_cls/model_cls:Class: 自定义配置和模型类型
- load_model_fn:Callable(model, model_args, **kwargs): 自定义模型加载，包括预训练参数等
- get_dataset_fn:Callable(train_args, model_args, data_args, eval_args, toeknizer, **kwargs): 定义数据加载和处理
- postprocess_dataset:Callable(examples, **kwargs): 自定义数据集后处理函数

### 评估(Evaluation)
-- MetricBase:Class: 自定义指标父类
-- BenchmarkBase-evaluate execute:Callable(input_ids, model, *args, **kwargs): 自定义评估过程

### 推理(Inference)

## 待定(To be continued) 
-- Inference
-- Evaluation
-- Supports for more datasets/benchmarks/models

