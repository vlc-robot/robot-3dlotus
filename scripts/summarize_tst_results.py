import os
import numpy as np
import json
import jsonlines
import collections
import argparse


split_names = ['taskvars_train', 'taskvars_test_l2', 'taskvars_test_l3', 'taskvars_test_l4']

def main(args):
    

    results = collections.defaultdict(list)

    for seed in args.seeds:
        print('load seed', seed)
        result_file = os.path.join(args.result_dir, f'seed{seed}', 'results.jsonl')
        if os.path.exists(result_file):
            with jsonlines.open(result_file, 'r') as f:
                for item in f:
                    if isinstance(item['checkpoint'], int):
                        res_ckpt = item['checkpoint']
                    else:
                        res_ckpt = int(os.path.basename(item['checkpoint']).split('_')[-1].split('.')[0])
                    if res_ckpt == args.ckpt_step:
                        taskvar = f"{item['task']}+{item['variation']}"
                        results[taskvar].append(item['sr'])
        else:
            print(result_file, 'missing')

    for split_name in split_names:
        print('split', split_name)
        taskvars = json.load(open(os.path.join('assets', f'{split_name}.json')))
        taskvars.sort()
        taskvars_sr = [np.mean(results[taskvar])*100 for taskvar in taskvars]
        taskvars_std = [np.std(results[taskvar])*100 for taskvar in taskvars]
        print(','.join(['avg'] + taskvars))
        print(','.join([f'{x:.2f}' for x in [np.mean(taskvars_sr)] + taskvars_sr])) 
        print(','.join([f'{x:.2f}' for x in [np.mean(taskvars_std)] + taskvars_std])) 
        
        num_seeds = min([len(results[taskvar]) for taskvar in taskvars])
        print('performance over seeds', num_seeds)
        seed_results = [100*np.mean([results[taskvar][i] for taskvar in taskvars]) \
            for i in range(min(len(args.seeds), num_seeds))]
        print(np.mean(seed_results), np.std(seed_results))
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir')
    parser.add_argument('ckpt_step', type=int)
    parser.add_argument('--seeds', type=int, nargs='+', default=[200, 300, 400, 500, 600])
    args = parser.parse_args()
    
    main(args)
