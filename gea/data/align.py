from typing import Callable, Dict, List, Any

def align_text(examples: Dict[str, Any], contexts: str) -> Dict[str, str]:
    return { "contexts": examples[contexts] }

def align_dialogue(examples: Dict[str, Any], 
                   contexts: List[str], 
                   instructions: List[str], 
                   responses: List[str], 
                   conversations: List[str] = None,
                   roles: List[str] = None) -> Dict[str, List[str]]:
    _contexts, _instructions, _responses = [], [], []
    if roles is None:
        if contexts is not None:
            _contexts = [examples[ctx] for ctx in contexts]
        if instructions is not None:
            _instructions = [examples[instr] for instr in instructions]
        if responses is not None:
            _responses = [examples[resp] for resp in responses]
    else:
        if contexts is not None:
            for ctx in contexts:
                _contexts += [item[conversations[0]] for item in examples[ctx] if item[conversations[1]] == roles[0]]
        if instructions is not None:
            for instr in instructions:
                _instructions += [item[conversations[0]] for item in examples[instr] if item[conversations[1]] == roles[0]]
        if responses is not None:
            for resp in responses:
                _responses += [item[conversations[0]] for item in examples[resp] if item[conversations[1]] == roles[1]]
    return { "contexts":_contexts, "instructions":_instructions, "responses":_responses }

def align_multi_turn(examples: Dict[str, Any], 
                   contexts: List[str], 
                   instructions: List[str], 
                   responses: List[str], 
                   conversations: List[str] = None,
                   roles: List[str] = None) -> Dict[str, List[str]]:
    if contexts is not None:
        _contexts = []
        for ctx in contexts:
            _contexts += [item[conversations[0]] for item in examples[ctx] if item[conversations[1]] == roles[0]]
    if instructions is not None:
        _instructions = []
        for instr in instructions:
            _instructions += [item[conversations[0]] for item in examples[instr] if item[conversations[1]] == roles[0]]
    if responses is not None:
        _responses = []
        for resp in responses:
            _responses += [item[conversations[0]] for item in examples[resp] if item[conversations[1]] == roles[1]]
    return { "contexts":_contexts, "instructions":_instructions, "responses":_responses }

ALIGN_FUNCTIONS: Dict[str, Dict[str, Callable]] = {
    "sequence": {
        "text": align_text,
        "dialogue": align_dialogue,
        "multi-turn": align_multi_turn,
    }

}