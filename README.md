<h1 align="center">General Execute Architecture</h1>

<p align="center">
  <img src="img/gea_logo.png" alt="GEA" style="width: 100%; max-width: 800px;">
</p>

## 简介 (Introduction)
作为一种通用的通用模型运行工具，包括训练、评估以及推理等任务，还在完善中，主要是我平时按需求添加功能。

## 特性 (Features)
1. 主要是提供更加统一、自由度更高的规划模式，更快捷、方便地调整以及规划包括数据、训练、评估以及推理等任务。
2. 为用户提供更加统一、完备的实验环境，将包括数据处理、规划训练、训练中/后续评估、推理以及设计评价指标等内容整合进一个项目中。
3. 自定义内容都提供默认执行方法。

### 训练 (Training)
- `execute_train_process`: Callable(accelerator, model, optimizer, lr_scheduler, train_dataloader, eval_dataloaders, train_state, callback_handler, **kwargs): 规划训练过程。
- `compute_loss`: Callable(model, batch, state, **kwargs): 规划 model & inputs 计算得到 loss & metrics。
- `execute_metrics`: Callbale(model, batch, state, **kwargs): 用于训练过程中自定义评估的loss & metrics
- `prepare_inputs_fn`: Callable(inputs, **kwargs): 处理 model 即将接受的数据。
- `exec_eval_fn`: Callable(model, eval_dataloader, train_state, description, **kwargs): 自定义训练中评估过程。
- `run_train`: Callable(**kwargs)：传入自定义参数。
- `lr_scheduler_kwargs`: learning rate scheduler的自定义参数。
- `inputs_kwargs`: 在`compute_loss`中传入的自定义输入参数。

### 模型和数据 (Models & Datas)
- `get_model_fn`: Callable(model_args, **kwargs): 定义模型和 tokenizer。
- `config_cls/model_cls`: Class: 自定义配置和模型类型。
- `load_model_fn`: Callable(model, model_args, **kwargs): 自定义模型加载，包括预训练参数等。
- `get_dataset_fn`: Callable(train_args, model_args, data_args, eval_args, tokenizer, **kwargs): 定义数据加载和处理。
- `postprocess_dataset`: Callable(examples, **kwargs): 自定义数据集后处理函数。
- `train_collate_fn` & `eval_collate_fn`: 自定义dataloader collate function。
- `model_kwargs`: 关于模型的自定义超参数。
- `data_kwargs`: 处理数据的自定义超参数。

### 评估 (Evaluation)
- `MetricBase`: Class: 自定义指标父类。
- `BenchmarkBase-evaluate execute`: Callable(input_ids, model, *args, **kwargs): 自定义评估过程。

### 推理 (Inference)

## 使用（Usage）
1. 创建scripts类似于example.sh
2. 创建executeor类似于gea/self/example下的文件：accelerate_config.yaml(deepspeed_config.json)和gpt_executor.py(modeling_gpt.py)

## 待定 (To be continued) 
- Inference
- Evaluation
- Supports for more datasets/benchmarks/models
