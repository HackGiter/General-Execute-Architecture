from functools import partial
from dataclasses import dataclass
from typing import Callable, Literal, Tuple, Union, Dict, List, Any

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel
from fastchat.model import get_conversation_template

from .experiment import Experiments
from ..model.template import Template
from ..data.profile import get_prompts
from ..metric.metrics import MetricBase

def gsm8k_fastchat(examples: Dict[str, Any], tokenizer: AutoTokenizer, max_length:int = 2048, template:Template=None) -> Dict[str, torch.Tensor]:
    conversations = examples['question'].strip()
    template = get_conversation_template(template.name)
    template.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    roles = {"human":template.roles[0], "gpt":template.roles[1]}
    template.messages = [
        [
            roles['human'],
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
        ],
        [
            roles['gpt'],
            "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."
        ],
        [
            roles['human'],
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"
        ],
        [
            roles['gpt'],
            "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."
        ],
        [
            roles['human'],
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"
        ],
        [
            roles['gpt'],
            "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."
        ],
        [
            roles['human'],
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"
        ],
        [
            roles['gpt'],
            "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."
        ],
        [
            roles['human'],
            "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"
        ],
        [
            roles['gpt'],
            "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."
        ],
        [
            roles['human'],
            "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"
        ],
        [
            roles['gpt'],
            " There were originally 9 computers. For each of 4 days, 5 more computerswere added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."
        ],
        [
            roles['human'],
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"
        ],
        [
            roles['gpt'],
            "Michael started with 58 golf balls. After losing 23 on tuesday, he had58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."
        ],
        [
            roles['human'],
            "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"
        ],
        [
            roles['gpt'],
            "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."
        ],
    ]
    template.append_message(roles['human'], conversations)
    conversations = template.get_prompt() + "[/INST]"
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
    )
    return {"input_ids":input_ids['input_ids'], "attention_mask":input_ids['attention_mask']}

def humaneval_fastchat(examples: Dict[str, Any], tokenizer: AutoTokenizer, max_length:int = 2048, template:Template=None) -> Dict[str, torch.Tensor]:
    conversations = examples['instruction'].strip()
    template = get_conversation_template(template.name)
    template.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    roles = {"human":template.roles[0], "gpt":template.roles[1]}
    template.messages = []
    template.append_message(roles['human'], conversations)
    conversations = template.get_prompt() + "[/INST] " + examples['prompt'].strip()
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
    )
    return {"input_ids":input_ids['input_ids'], "attention_mask":input_ids['attention_mask']}

def mt_bench_fastchat(examples: Dict[str, Any], tokenizer: AutoTokenizer, max_length:int = 2048, template:Template=None) -> Dict[str, Any]:
    category = examples['category']
    turns = examples['turns']
    template = get_conversation_template(template.name)
    template.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    roles = {"human":template.roles[0], "gpt":template.roles[1]}
    template.messages = []
    template.append_message(roles['human'], turns[0].strip())
    conversations = template.get_prompt() + "[/INST]"
    input_ids = [tokenizer(
        conversations,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
    )]
    for turn in turns[1:]:
        template.system_message = None
        template.messages = []
        template.append_message(roles['human'], turn.strip())
        conversations = template.get_prompt() + "[/INST]"
        input_ids += [tokenizer(
            conversations,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )]
    return {"input_ids":input_ids['input_ids'], "attention_mask":input_ids['attention_mask'], "category": category}

def gsm8k(examples: Dict[str, Any], tokenizer: AutoTokenizer, sys_prompt:str=None, max_length:int=2048, template:Template=None) -> Dict[str, torch.Tensor]:
    inputs = [
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        examples['question'],
    ]
    labels = [
        "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
        " There were originally 9 computers. For each of 4 days, 5 more computerswere added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ]
    examples = {
        "instructions":inputs,
        "responses": labels,
    }
    return get_prompts(examples, sys_prompt, tokenizer, max_length, template, True)

def humaneval(examples: Dict[str, Any], tokenizer: AutoTokenizer, sys_prompt:str=None, max_length:int=2048, template:Template=None) -> Dict[str, torch.Tensor]:
    examples = {
        "instructions": [examples['instruction']],
        "responses": [examples['prompt']]
    }
    return get_prompts(examples, sys_prompt, tokenizer, max_length, template, False)

def mt_bench(examples: Dict[str, Any], tokenizer: AutoTokenizer, sys_prompt:str=None, max_length:int=2048, template:Template=None) -> Dict[str, Any]:
    inputs = {
        "instructions":examples['turns'],
        "responses":[]
    }
    outputs = get_prompts(inputs, sys_prompt, tokenizer, max_length, template, True, False)
    outputs['categories'] = [examples['category']]
    return outputs

@dataclass
class SequenceOutputs:
    sequences:torch.Tensor = None
    past_key_values: Tuple[torch.Tensor] = None
    metrics: MetricBase = None
    kwargs: Dict[str, Any] = None

    def __post_init__(self):
        self.kwargs = {}

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key):
        # 允许使用 obj[key] 访问属性
        return getattr(self, key)
    
    def __iter__(self):
        # 使实例可以直接通过 ** 解包为字典
        for key, value in self.__dict__.items():
            yield key, value

class BenchmarkBase:
    name: str = None
    path: str = None
    split: str = None
    process: Callable = None
    tokenizer: AutoTokenizer = None
    template: Template = None
    max_length: int = None
    sys_prompt: str = None
    column_names: Union[List[str], Tuple[str]] = None

    def __init__(self, 
                 name:str = None, 
                 path:str = None,
                 split:str = None, 
                 process:Callable = None, 
                 tokenizer:AutoTokenizer = None,
                 template:Template = None,
                 **kwagrs) -> None:
        self.name = name
        self.path = path
        self.split = split
        self.process = process
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = kwagrs.get("max_length", None)
        self.sys_prompt = kwagrs.get("sys_prompt", template.sys_prompt)

    def prepare(self, max_length:int=None, sys_prompt:str=None):
        if max_length is not None:
            self.max_length = max_length
        if sys_prompt is not None:
            self.sys_prompt = sys_prompt
        self._dataset = load_dataset(
            path=self.path,
            name=self.split,
            trust_remote_code=True,
        )
        if not isinstance(self._dataset, Dataset):
            keys = list(self._dataset.keys())
            self._dataset = self._dataset['test'] if 'test' in keys else self._dataset[keys[0]]
        self.column_names = self._dataset.column_names
        self._dataset = self._dataset.map(
            partial(self.process, tokenizer=self.tokenizer, template=self.template, sys_prompt=self.sys_prompt, max_length=self.max_length),
            num_proc=4,
            remove_columns=self.column_names,
        )
        self._dataset.set_format("torch")

    def evaluate(self, exp:Experiments, excecute:Callable, model:AutoModel, *args):
        NotImplemented

class GSM8K(BenchmarkBase):
    def __init__(self, 
                 path: str = '/data/lhz/datasets/gsm8k_2/', 
                 use_fastchat: bool = False,
                 tokenizer: AutoTokenizer = None, **kwargs) -> None:
        super().__init__("GSM8K", path, "main", gsm8k_fastchat if use_fastchat else gsm8k , tokenizer, **kwargs)

    def evaluate(self, exp:Experiments, excecute:Callable, model:AutoModel, *args):
        if getattr(self, "_dataset", None) is None:
            self.prepare()
        torch.cuda.empty_cache()
        exp.tqdm("INFERENCING", len(self._dataset))
        kwargs = {**exp}
        for _, item in enumerate(self._dataset):
            input_ids = item['input_ids'].to(model.device)
            outputs:SequenceOutputs = excecute(
                input_ids,
                model,
                *args,
                **kwargs,
            )
            if exp.add(outputs):
                break
        exp.summary()

class HUMANEVAL(BenchmarkBase):
    def __init__(self, 
                 path: str = '/data/lhz/datasets/humanevalpack/',
                 use_fastchat: bool = False,
                 tokenizer: AutoTokenizer = None, **kwargs) -> None:
        super().__init__("HUMANEVAL", path, "python", humaneval_fastchat if use_fastchat else humaneval, tokenizer, **kwargs)

    def evaluate(self, exp:Experiments, excecute:Callable, model:AutoModel, *args):
        torch.cuda.empty_cache()
        exp.tqdm("INFERENCING", len(self._dataset))
        kwargs = {**exp}
        for _, item in enumerate(self._dataset):
            input_ids = item['input_ids'].to(model.device)
            outputs:SequenceOutputs = excecute(
                input_ids,
                model,
                *args,
                **kwargs,
            )
            if exp.add(outputs):
                break
        exp.summary()

class MT_BENCH(BenchmarkBase):
    def __init__(self, 
                 path: str = '/data/lhz/datasets/mt-bench/',
                 use_fastchat: bool = False, 
                 tokenizer: AutoTokenizer = None, **kwargs) -> None:
        super().__init__("MT-BENCH", path, None, mt_bench_fastchat if use_fastchat else mt_bench, tokenizer, **kwargs)

    def evaluate(self, exp:Experiments, excecute:Callable, model:AutoModel, *args, **kwargs):
        torch.cuda.empty_cache()
        sequential:bool = kwargs.pop("sequential", False)
        exp.tqdm("INFERENCING", len(self._dataset))
        kwargs = { **exp }
        for _, items in enumerate(self._dataset):
            input_ids = None
            for item in items['input_ids']:
                input_ids = item.to(model.device) if input_ids is None else torch.cat([input_ids, item.to(model.device)], dim=-1)
                outputs:SequenceOutputs = excecute(
                    input_ids,
                    model,
                    *args,
                    **kwargs,
                )
                input_ids = outputs.sequences
                kwargs = { **outputs.kwargs, **exp } if sequential else kwargs
                exp.merge(outputs)
            if exp.add(outputs):
                break
        exp.summary()

BENCHMARK_CATEGORIES: Dict[str, BenchmarkBase] = {
    'GSM8K': GSM8K,
    'HUMANEVAL': HUMANEVAL,
    'MT-BENCH': MT_BENCH,
}

def get_benchmark(
    benchmark: Literal['GSM8K', 'HUMANEVAL', 'MT-BENCH'],
    use_fastchat: bool = False,
    tokenizer: AutoTokenizer = None,
    **kwargs,
) -> BenchmarkBase:
    bc: BenchmarkBase = BENCHMARK_CATEGORIES[benchmark]
    return bc(use_fastchat=use_fastchat, tokenizer=tokenizer, **kwargs)