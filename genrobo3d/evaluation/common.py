import torch

import jsonlines
from filelock import FileLock
import re
from easydict import EasyDict


def write_to_file(filepath, data):
    lock = FileLock(filepath+'.lock')
    with lock:
        with jsonlines.open(filepath, 'a', flush=True) as outf:
            outf.write(data)


def load_checkpoint(model, ckpt_file):
    ckpt = torch.load(ckpt_file)
    state_dict = model.state_dict()
    new_ckpt = {}
    for k, v in ckpt.items():
        if k in state_dict:
            new_ckpt[k] = v
        else:
            print(k, 'not in model')
    for k, v in state_dict.items():
        if k not in new_ckpt:
            print(k, 'not in ckpt')
    model.load_state_dict(new_ckpt, strict=True)


def parse_code(code):
    pattern = re.compile(r'^((?P<ret_val>\w+) = ){0,1}(?P<action>\w+)\((object=(?P<object>[\w\s"\']+)){0,1}(,\s){0,1}(target=(?P<target>[\w\s"\']+)){0,1}(,\s){0,1}(not=\[(?P<not_objects>[\w\s"\',]+)\]){0,1}\)')

    res = re.search(pattern, code)

    if res is None or res['action'] is None:
        print('invalid code', code)
        return None
    
    action_name = res['action'].replace('_', ' ')
    
    not_objects = None
    if res['not_objects'] is not None:
        not_objects = [x.strip() for x in res['not_objects'].split(',')]   # always variable

    object_name, is_object_variable = None, False
    if res['object'] is not None:
        if res['object'][0] == res['object'][-1] and res['object'][0] in ['"', '\'']:
            object_name = res['object'][1:-1]
        else:
            object_name = res['object']
            is_object_variable = True

    target_name, is_target_variable = None, False
    if res['target'] is not None:
        if res['target'][0] == res['target'][-1] and res['target'][0] in ['"', '\'']:
            target_name = res['target'][1:-1]
        else:
            target_name = res['target']
            is_target_variable = True

        if target_name in ['up', 'out', 'down']:
            action_name = ' '.join([action_name, target_name])
            target_name = None

    return EasyDict(
        action=action_name, 
        object=object_name, target=target_name, 
        is_target_variable=is_target_variable, is_object_variable=is_object_variable,
        not_objects=not_objects, ret_val=res['ret_val'],
    )

