from typing import Callable, Dict, List, Any

from ..utils.logging import get_logger

logger = get_logger(__name__)

def align_text(examples: Dict[str, List[Any]], contexts: str, **kwargs) -> Dict[str, str]:
    return { "contexts": examples[contexts] }

def align_dialogue(examples: Dict[str, List[Any]], 
                   contexts: List[str], 
                   instructions: List[str], 
                   responses: List[str], **kwargs) -> Dict[str, List[str]]:
    _contexts, _instructions, _responses = [], [], []
    for key, value in examples.items():
        if key in instructions:
            _instructions += ([[] for _ in range(len(value))] if len(_instructions) == 0 else [])
            for i, item in enumerate(value):
                _instructions[i].append(item)
        elif key in responses:
            _responses += ([[] for _ in range(len(value))] if len(_responses) == 0 else [])
            for i, item in enumerate(value):
                _responses[i].append(item)
        elif key in contexts:
            _contexts += ([[] for _ in range(len(value))] if len(_contexts) == 0 else [])
            for i, item in enumerate(value):
                _contexts[i].append(item)
    examples = {}
    if len(_contexts) > 0:
        examples["contexts"] = _contexts
    if len(_instructions) > 0:
        examples["instructions"] = _instructions
    if len(_responses) > 0:
        examples["responses"] = _responses
    return examples

def align_multi_turn(examples: Dict[str, List[Any]], 
                   contexts: List[str], 
                   instructions: List[str], 
                   responses: List[str], 
                   conversations: List[str] = None,
                   roles: List[str] = None, **kwargs) -> Dict[str, List[str]]:
    _contexts, _instructions, _responses = [], [], []
    for key, value in examples.items():
        if key in instructions:
            _instructions += ([[] for _ in range(len(value))]if len(_instructions) == 0 else [])
            for i, v in enumerate(value):
                _instructions[i] += [item[conversations[0]] for item in v if item[conversations[1]] == roles[0]]
        elif key in responses:
            _responses += ([[] for _ in range(len(value))] if len(_responses) == 0 else [])
            for i, v in enumerate(value):
                _responses[i] += [item[conversations[0]] for item in v if item[conversations[1]] == roles[0]]
        elif key in contexts:
            _contexts += ([[] for _ in range(len(value))] if len(_contexts) == 0 else [])
            for i, v in enumerate(value):
                _contexts[i] += [item[conversations[0]] for item in v if item[conversations[1]] == roles[0]]
    examples = {}
    if len(_contexts) > 0:
        examples["contexts"] = _contexts
    if len(_instructions) > 0:
        examples["instructions"] = _instructions
    if len(_responses) > 0:
        examples["responses"] = _responses
    return examples

ALIGN_FUNCTIONS: Dict[str, Dict[str, Callable]] = {
    "sequence": {
        "text": align_text,
        "dialogue": align_dialogue,
        "multi-turn": align_multi_turn,
    }

}