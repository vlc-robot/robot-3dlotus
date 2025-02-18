import os
import json
import numpy as np

from tqdm import tqdm

from genrobo3d.vlm_models.clip_encoder import ClipEncoder, OpenClipEncoder

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, default='assets/taskvars_instructions_new.json')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_name', default='clip', choices=['openclip', 'clip'])

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'instr_embeds_{args.model_name}.npy')
    if os.path.exists(output_file):
        return
    
    taskvars_instrs = json.load(open(args.input_file))
    all_instrs = set()
    for instrs in taskvars_instrs.values():
        for instr in instrs:
            all_instrs.add(instr)
    print(len(all_instrs))

    if args.model_name == 'clip':
        clip_model = ClipEncoder()
    elif args.model_name == 'openclip':
        clip_model = OpenClipEncoder()
    
    action_embeds = {}
    for query in all_instrs:
        txt_embed = clip_model('text', query, use_prompt=False, output_hidden_states=True)
        txt_embed = txt_embed[0].data.cpu().numpy()
        action_embeds[query] = txt_embed

    np.save(output_file, action_embeds)

 
if __name__ == '__main__':
    main()
