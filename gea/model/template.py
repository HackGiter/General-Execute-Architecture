from dataclasses import dataclass
from typing import Callable, Tuple, Union, List, Dict, Any

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..args import ModelArguments
from ..utils.tools import get_logger, get_model_details

logger = get_logger(__name__)

IGNORE_INDEX=-100

@dataclass
class Template:
    name: str = None
    sys_tokens: List[str] = None
    inst_tokens: List[str] = None
    bos_token: str = None
    eos_token: str = None
    sys_prompt: str = None
    ignore_index: int = IGNORE_INDEX
    sys_template: Callable = None
    inst_template: Callable = None
    resp_template: Callable = None

    def get_sys_prompt(self, sys_prompt:str) -> str:
        return self.sys_template(sys_prompt, self.sys_tokens, self.bos_token, self.eos_token)
    
    def get_inst_prompt(self, instruction:str, sys_prompt:str) -> str:
        return self.inst_template(instruction, self.inst_tokens, sys_prompt, self.bos_token, self.eos_token)
    
    def get_response(self, response:str) -> str:
        return self.resp_template(response, self.inst_tokens, self.bos_token, self.eos_token)
    
    def get_prompts(
            self, 
            example:Dict[str, Any],
            sys_prompt:str,
            tokenizer:AutoTokenizer,
            max_length:int,
            eos_last:bool = True,
            concatenated: bool = True) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        instructions: List[str] = example["instructions"]
        responses: List[str] = example["responses"]
        sys_prompt = "" if sys_prompt is None else self.get_sys_prompt(sys_prompt)
        instructions = [
            (self.bos_token + self.get_inst_prompt(item.strip(), sys_prompt)) 
            if i == 0 else self.get_inst_prompt(item.strip(), "")
            for i, item in enumerate(instructions)
        ]
        responses = [
            item.lstrip() if not eos_last and i == len(responses) - 1 else self.get_response(item.strip())
            for i, item in enumerate(responses)
        ]
        instructions = [
            tokenizer(
                item, 
                return_tensors="pt",
                add_special_tokens=False
            ) for item in instructions
        ]
        responses = [
            tokenizer(
                item,
                return_tensors="pt",
                add_special_tokens=False
            ) for item in responses
        ]
        if len(instructions) != 0 and len(responses) != 0:
            input_ids = torch.cat(
                [torch.cat((instr["input_ids"], label["input_ids"]), dim=-1) for instr, label in zip(instructions, responses)] + 
                ([] if len(instructions) == len(responses) else 
                ([item["input_ids"] for item in responses[len(instructions):]] if len(instructions) < len(responses) else 
                [item["input_ids"] for item in instructions[len(responses):]]))
            , dim=-1)
            labels = torch.cat(
                [torch.cat((torch.full_like(instr["input_ids"], fill_value=self.ignore_index), label["input_ids"]), dim=-1) for instr, label in zip(instructions, responses)] + 
                ([] if len(instructions) == len(responses) else 
                ([item["input_ids"] for item in responses[len(instructions):]] if len(instructions) < len(responses) else 
                [item["input_ids"] for item in instructions[len(responses):]]))
            , dim=-1)
            attention_mask = torch.cat(
                [torch.cat((instr["attention_mask"], label["attention_mask"]), dim=-1) for instr, label in zip(instructions, responses)] + 
                ([] if len(instructions) == len(responses) else 
                ([item["attention_mask"] for item in responses[len(instructions):]] if len(instructions) < len(responses) else 
                [item["attention_mask"] for item in instructions[len(responses):]]))
            , dim=-1)
            if max_length is not None:
                input_ids = input_ids[:, :max_length]
                labels = labels[:, :max_length]
                attention_mask = attention_mask[:, :max_length]
        else:
            instructions = instructions if len(instructions) != 0 else responses
            if concatenated:
                input_ids = torch.cat([item["input_ids"] for item in instructions], dim=-1)
                labels = input_ids
                attention_mask = torch.cat([item["attention_mask"] for item in instructions], dim=-1)
            else:
                input_ids = [item["input_ids"] for item in instructions]
                labels = None
                attention_mask = [item["attention_mask"] for item in instructions]

        return { 
            "input_ids":input_ids, 
            "labels":labels, 
            "attention_mask":attention_mask 
            }
            
def get_llama2_prompts(
        self, 
        example:Dict[str, Any],
        sys_prompt:str,
        tokenizer:AutoTokenizer,
        max_length:int,
        eos_last:bool = True,
        concatenated: bool = True) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
    instructions: List[str] = example["instructions"]
    responses: List[str] = example["responses"]
    sys_prompt = "" if sys_prompt is None else self.get_sys_prompt(sys_prompt)
    instructions = [
        self.bos_token + self.get_inst_prompt(item.strip(), sys_prompt) 
        if i == 0 else self.get_inst_prompt(item.strip(), "")
        for i, item in enumerate(instructions)
    ]
    responses = [
        item.lstrip() if not eos_last and i == len(responses) - 1 else self.get_response(item.strip())
        for i, item in enumerate(responses)
    ]
    instructions = [
        tokenizer(
            item, 
            return_tensors="pt",
            add_special_tokens=False
        ) for item in instructions
    ]
    responses = [
        tokenizer(
            item,
            return_tensors="pt",
            add_special_tokens=False
        ) for item in responses
    ]
    if len(instructions) != 0 and len(responses) != 0:
        input_ids = torch.cat(
            [torch.cat((instr["input_ids"], label["input_ids"]), dim=-1) for instr, label in zip(instructions, responses)] + 
            ([] if len(instructions) == len(responses) else 
            ([item["input_ids"] for item in responses[len(instructions):]] if len(instructions) < len(responses) else 
            [item["input_ids"] for item in instructions[len(responses):]]))
        , dim=-1)
        labels = torch.cat(
            [torch.cat((torch.full_like(instr["input_ids"], fill_value=self.ignore_index), label["input_ids"]), dim=-1) for instr, label in zip(instructions, responses)] + 
            ([] if len(instructions) == len(responses) else 
            ([item["input_ids"] for item in responses[len(instructions):]] if len(instructions) < len(responses) else 
            [item["input_ids"] for item in instructions[len(responses):]]))
        , dim=-1)
        attention_mask = torch.cat(
            [torch.cat((instr["attention_mask"], label["attention_mask"]), dim=-1) for instr, label in zip(instructions, responses)] + 
            ([] if len(instructions) == len(responses) else 
            ([item["attention_mask"] for item in responses[len(instructions):]] if len(instructions) < len(responses) else 
            [item["attention_mask"] for item in instructions[len(responses):]]))
        , dim=-1)
        if max_length is not None:
            input_ids = input_ids[:, :max_length]
            labels = labels[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
    else:
        instructions = instructions if len(instructions) != 0 else responses
        if concatenated:
            input_ids = torch.cat([item["input_ids"] for item in instructions], dim=-1)
            labels = input_ids
            attention_mask = torch.cat([item["attention_mask"] for item in instructions], dim=-1)
        else:
            input_ids = [item["input_ids"] for item in instructions]
            labels = None
            attention_mask = [item["attention_mask"] for item in instructions]

    return { "input_ids":input_ids, "labels":labels, "attention_mask":attention_mask }

MODEL_TEMPLATES:Dict[str, Template] = {}

def register_template(
    name: str = None,
    sys_tokens: List[str] = None,
    inst_tokens: List[str] = None,
    bos_token: str = None,
    eos_token: str = None,
    sys_prompt: str = None,
    ignore_index: int = IGNORE_INDEX,
    sys_template: Callable = None,
    inst_template: Callable = None,
    resp_template: Callable = None,
    get_prompts: Callable = None,
) -> None:
    MODEL_TEMPLATES[name] = Template(
        name=name,
        sys_tokens=sys_tokens,
        inst_tokens=inst_tokens,
        bos_token=bos_token,
        eos_token=eos_token,
        sys_prompt=sys_prompt,
        ignore_index=ignore_index,
        sys_template=sys_template,
        inst_template=inst_template,
        resp_template=resp_template,
    )
    if get_prompts is not None:
        MODEL_TEMPLATES[name].get_prompts = get_prompts

register_template(
    name="llama2",
    sys_tokens=["<<SYS>>", "<</SYS>>"],
    inst_tokens=["[INST] ", " [/INST]"],
    bos_token="<s>",
    eos_token="</s>",
    sys_prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
    sys_template=lambda sys_prompt, sys_tokens, bos_token, eos_token: 
    f"{sys_tokens[0]}\n{sys_prompt}\n{sys_tokens[1]}\n\n",
    
    inst_template=lambda instruction, inst_tokens, sys_prompt, bos_token, eos_token: 
    f"{bos_token}{inst_tokens[0]}{sys_prompt}{instruction}{inst_tokens[1]}",
    
    resp_template=lambda response, inst_tokens, bos_token, eos_token: 
    f"{response} {eos_token}",
    get_prompts=get_llama2_prompts
)

register_template(
    name="llama3",
    sys_tokens=["<|start_header_id|>", "<|end_header_id|>"],
    inst_tokens=["<|start_header_id|>", "<|end_header_id|>"],
    bos_token="<|begin_of_text|>",
    eos_token="<|eot_id|>",
    sys_prompt="Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n",
    sys_template=lambda sys_prompt, sys_tokens, bos_token, eos_token: 
    f"{sys_tokens[0]}system{sys_tokens[1]}{sys_prompt}{eos_token}",
    
    inst_template=lambda instruction, inst_tokens, sys_prompt, bos_token, eos_token: 
    f"{bos_token}{sys_prompt}{inst_tokens[0]}user{inst_tokens[1]}{instruction}{eos_token}",
    
    resp_template=lambda response, inst_tokens, bos_token, eos_token: 
    f"{inst_tokens[0]}assistant{inst_tokens[1]}{response}{eos_token}"
)

register_template(
    name="empty",
    sys_tokens=[],
    inst_tokens=[],
    bos_token="",
    eos_token="",
    sys_prompt=None,
    sys_template=lambda sys_prompt, sys_tokens, bos_token, eos_token: "",
    inst_template=lambda instruction, inst_tokens, sys_prompt, bos_token, eos_token: f"{instruction}",
    resp_template=lambda response, bos_token, eos_token: f"{response}"
)

def get_model(
        model_args:ModelArguments, 
        **kwargs) -> Tuple[AutoModel, AutoTokenizer]:
    """
    set `load_model_fn` function if you want to load your model in your own way such as load with parts of pretrained way
    """
    logger.info("Model initialize")
    # model_kwargs = kwargs.pop("model_kwargs", {})
    load_model_fn:Callable = kwargs.pop("load_model_fn", None)

    if model_args.pretrained:
        tokenizer = AutoTokenizer.from_pretrained(model_args.path)
        model = AutoModel.from_pretrained(model_args.path, **kwargs)
    else:
        pretrained_model_path = kwargs.pop("pretrained_model_path", None)
        if pretrained_model_path is None:       
            tokenizer = AutoTokenizer.from_pretrained(model_args.path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        config_cls = kwargs.pop("config_cls", AutoConfig)
        model_cls = kwargs.pop("model_cls", AutoModel)
        config = config_cls.from_pretrained(model_args.path, **kwargs)
        model = model_cls(config, **kwargs)
        
        if load_model_fn is not None:
            model = load_model_fn(model, model_args=model_args, pretrained_model_path=pretrained_model_path, **kwargs)
            
    logger.info(f"\n{get_model_details(model, True)}")
    return (model, tokenizer, )



