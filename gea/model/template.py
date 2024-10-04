from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple

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
        return self.resp_template(response, self.bos_token, self.eos_token)
    
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
):
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

register_template(
    name="llama2",
    sys_tokens=['<<SYS>>', '<</SYS>>'],
    inst_tokens=['[INST] ', ' [/INST]'],
    bos_token='<s>',
    eos_token='</s>',
    sys_prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
    sys_template=lambda sys_prompt, sys_tokens, bos_token, eos_token: f"{sys_tokens[0]}\n{sys_prompt}\n{sys_tokens[1]}\n\n",
    inst_template=lambda instruction, inst_tokens, sys_prompt, bos_token, eos_token: f"{bos_token}{inst_tokens[0]}{sys_prompt}{instruction}{inst_tokens[1]}",
    resp_template=lambda response, bos_token, eos_token: f"{response} {eos_token}"
)

register_template(
    name="llama3",
    sys_tokens=['<<SYS>>', '<</SYS>>'],
    inst_tokens=['[INST] ', ' [/INST]'],
    bos_token='<s>',
    eos_token='</s>',
    sys_prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
    sys_template=lambda sys_prompt, sys_tokens, bos_token, eos_token: f"{sys_tokens[0]}\n{sys_prompt}\n{sys_tokens[1]}\n\n",
    inst_template=lambda instruction, inst_tokens, sys_prompt, bos_token, eos_token: f"{bos_token}{inst_tokens[0]}{sys_prompt}{instruction}{inst_tokens[1]}",
    resp_template=lambda response, bos_token, eos_token: f"{response} {eos_token}"
)

register_template(
    name="empty",
    sys_tokens=[],
    inst_tokens=[],
    bos_token='',
    eos_token='',
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



