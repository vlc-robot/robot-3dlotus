import os
import numpy as np
import jsonlines
import collections
import argparse

def main(args):
    results = collections.defaultdict(list)
    taskvar_set = set()
    with jsonlines.open(args.result_file, 'r') as f:
        for item in f:
            ckpt_step = int(os.path.basename(item['checkpoint']).split('.')[0].split('_')[-1]) if isinstance(item['checkpoint'], str) else item['checkpoint']
            if args.ckpt_step is not None and ckpt_step != args.ckpt_step:
                continue
            if (item['checkpoint'], item['task'], item['variation']) in taskvar_set:
                continue
            results[item['checkpoint']].append((item['task'], item['variation'], item['sr'], item['num_demos']))
            taskvar_set.add((item['checkpoint'], item['task'], item['variation']))

    ckpts = list(results.keys())
    if isinstance(ckpts[0], int):
        ckpts.sort()
    else:
        ckpts.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))

    # show task results
    taskvars = set()
    for ckpt in ckpts:
        for x in results[ckpt]:
            if args.aggr_task:
                taskvars.add(x[0])
            else:
                taskvars.add((x[0], x[1]))
    taskvars = list(taskvars)
    if args.aggr_task:
        taskvars.sort()
    else:
        taskvars.sort(key=lambda x: (x[0], x[1]))
    for taskvar in taskvars:
        res = []
        for ckpt in ckpts:
            ckpt_res = []
            for x in results[ckpt]:
                if args.aggr_task and x[0] == taskvar:
                    if args.sr_per_task:
                        ckpt_res.append((x[2]*x[3], x[3]))
                    else:
                        ckpt_res.append(x[2])
                if (not args.aggr_task) and x[0] == taskvar[0] and x[1] == taskvar[1]:
                    ckpt_res.append(x[2])
            if args.aggr_task and args.sr_per_task:
                res.append(np.sum([x[0] for x in ckpt_res]) / np.sum([x[1] for x in ckpt_res]))
            else:
                res.append(np.mean(ckpt_res))
        if args.aggr_task:
            print('\n', taskvar, len(ckpt_res))
        else:
            print('\n', f'{taskvar[0]}+{taskvar[1]}', len(ckpt_res))
        print(', '.join(['%.2f' % (x*100) for x in res]))
    print()

    avg_results = []
    for k in ckpts:
        v = results[k]
        if args.first_avg_task:
            if args.aggr_task:
                sr = collections.defaultdict(list)
                for x in v:
                    sr[x[0]].append([x[2] * x[3], x[3]])
                sr = [np.sum([c[0] for c in x])/np.sum([c[1] for c in x]) for x in sr.values()]
            else:
                sr = collections.defaultdict(list)
                for x in v:
                    sr[x[0]].append(x[2])
                sr = [np.mean(x) for x in sr.values()]
        else:
            sr = [x[2] for x in v]
        print(k, len(v), np.mean(sr)*100)
        avg_results.append((k, np.mean(sr)))

    print()
    print('Best checkpoint and SR')
    avg_results.sort(key=lambda x: -x[1])
    for x in avg_results:
        if x[-1] < avg_results[0][-1]:
            break
        print((x[0], x[1]*100))
    print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_file')
    parser.add_argument('--aggr_task', action='store_true', default=False)
    parser.add_argument('--first_avg_task', action='store_true', default=False)
    parser.add_argument('--sr_per_task', action='store_true', default=False)
    parser.add_argument('--ckpt_step', default=None, type=int)
    args = parser.parse_args()
    main(args)
