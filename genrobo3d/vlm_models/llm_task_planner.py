from typing import List, Optional

import os
import string
import json
import random
import einops
import jsonlines

import torch
import torch.nn.functional as F

from llama import Dialog, Llama
from transformers import AutoTokenizer, AutoModel


class LlamaTaskPlanner(object):
    def __init__(
        self, prompt_dir, asset_dir, 
        temperature=0, max_seq_len=8192, top_p=0.9, max_batch_size=1, 
        max_gen_len=None, device=None, master_port=12300,
        ckpt_dir=None, groq_model=None, cache_file=None,
    ):
        '''
        groq_models: https://console.groq.com/docs/models
        '''
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if ckpt_dir is not None:    # Load llama3 locally
            # if cache_file is None:  # TODO
            self.use_local_llama = True
            tokenizer_path = os.path.join(ckpt_dir, 'tokenizer.model')

            # # Llama can only be loaded in distributed env
            # os.environ['MASTER_ADDR'] = 'localhost'
            # os.environ['MASTER_PORT'] = str(master_port)
            init_method = "file:///tmp/tmpllama"
            # TODO: change the device
            if not torch.distributed.is_initialized():    
                torch.distributed.init_process_group(
                    backend='nccl', init_method=init_method, rank=0, world_size=1
                )
            # https://github.com/openai/tiktoken/issues/75
            os.environ['TIKTOKEN_CACHE_DIR'] = ""

            self.generator = Llama.build(
                ckpt_dir=ckpt_dir,
                tokenizer_path=tokenizer_path,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                device=self.device
            )
                # torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            from groq import Groq

            assert groq_model is not None
            self.generator = Groq(api_key=os.environ.get('GROQ_API_KEY'))
            self.groq_model_name = groq_model
            self.use_local_llama = False

        # Load sentence-bert to measure sentence similarity
        self.bert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.bert_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(self.device)

        self.temperature = temperature  # 0 means greedy decoding
        self.top_p = top_p              # for nucleus sampling
        self.max_seq_len = max_seq_len  # The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.
        self.max_gen_len = max_gen_len # `max_gen_len` is optional because finetuned models are able to stop generations naturally.

        self.prompt_dir = prompt_dir
        self.asset_dir = asset_dir

        self.load_prompts()
        self.load_in_context_examples()
        self.load_instruction_embeds()

        self.cache = {}
        if cache_file is not None:
            with jsonlines.open(cache_file, 'r') as f:
                for item in f:
                    plans = [line.strip() for line in item['results'].split('\n')]
                    plans = [line for line in plans if (not line.startswith('#')) and len(line) > 0]
                    self.cache[item['instruction']] = (item['results'], plans)

    def load_prompts(self):
        self.system_prompt = open(os.path.join(self.prompt_dir, 'system_prompt.txt')).readlines()[0].strip()
        self.user1 = ''.join(open(os.path.join(self.prompt_dir, 'planner_prompts.txt')).readlines())
        self.assistant1 = f'Got it. I will complete what you give me next.'

        self.height_range_prompts = json.load(
            open(os.path.join(self.prompt_dir, 'height_range_prompts.json'))
        )

    def load_in_context_examples(self):
        example_file = os.path.join(self.prompt_dir, 'in_context_examples.txt')
        data = [x.strip() for x in open(example_file, 'r').readlines() if len(x.strip()) > 0]
        taskvar_examples = {}
        for line in data:
            if line.startswith('# taskvar:'):
                taskvar = line.split('# taskvar:')[-1].strip()
                taskvar_examples.setdefault(taskvar, [])
                taskvar_examples[taskvar].append([])
            elif line.startswith('# query:'):
                # replace with other instructions of the taskvar in the future
                taskvar_examples[taskvar][-1].append('# query: {instruction}')
            else:
                taskvar_examples[taskvar][-1].append(line)
        self.taskvar_examples = taskvar_examples

    def load_instruction_embeds(self):
        trn_taskvars = set(json.load(open(os.path.join(self.asset_dir, 'taskvars_train.json'))))
        self.taskvar_instructions = {
            taskvar: [instr+'.' for instr in instrs] for taskvar, instrs in \
                json.load(open(os.path.join(self.asset_dir, 'taskvars_instructions_new.json'))).items() \
            if taskvar in trn_taskvars
        }

        self.instr_to_taskvar = {}
        for taskvar, instrs in self.taskvar_instructions.items():
            for instr in instrs:
                self.instr_to_taskvar[instr] = taskvar

        self.trn_instrs = list(self.instr_to_taskvar.keys())
        self.trn_instr_embeds = self.get_sentence_embeds(self.trn_instrs)

    @torch.no_grad
    def get_sentence_embeds(self, sentences):
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        encoded_input = self.bert_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        for k, v in encoded_input.items():
            encoded_input[k] = v.to(self.device)
        model_output = self.bert_model(**encoded_input)
        sentence_embeds = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeds = F.normalize(sentence_embeds, p=2, dim=1)
        return sentence_embeds
    
    @torch.no_grad
    def __call__(self, query, context=None, topk=20, verbose=False):
        if query in self.cache:
            return self.cache[query]

        if context is None and query in self.cache:
            return self.cache[query]

        if query[-1] not in string.punctuation:
            query = f'{query}.'
        user2 = f'# query: {query}'
        if context is not None:
            user2 = f'{user2}\n# objects = {context}'

        # select in-context examples
        query_sent_embeds = self.get_sentence_embeds([query])
        sims = einops.einsum(query_sent_embeds, self.trn_instr_embeds, 'q d, p d -> q p')[0]
        _, idxs = torch.sort(sims, dim=0, descending=True)
        topk_instrs = [self.trn_instrs[idx] for idx in idxs]
        topk_examples = []
        used_taskvars = set()
        for instr in topk_instrs:
            taskvar = self.instr_to_taskvar[instr]
            if taskvar not in used_taskvars:
                used_taskvars.add(taskvar)
                example = random.choice(self.taskvar_examples[taskvar])
                example_query = example[0].format(instruction=instr)
                if context is None:
                    topk_examples.append('\n'.join([example_query] + example[2:]))
                else:
                    topk_examples.append('\n'.join([example_query] + example[1:]))
            if len(topk_examples) >= topk:
                break
        topk_examples = '\n\n'.join(topk_examples[::])

        dialogs: List[Dialog] = [
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user1 + topk_examples},
                {"role": "assistant", "content": self.assistant1},
                {"role": "user", "content": user2}
            ],
        ]

        if self.use_local_llama:
            results = self.generator.chat_completion(
                dialogs,
                max_gen_len=self.max_gen_len,
                temperature=self.temperature,
                top_p=self.top_p,
                logprobs=False,
            )[0]['generation']['content']
        else:
            results = self.generator.chat.completions.create(
                messages=dialogs[0], 
                max_tokens=self.max_gen_len,
                temperature=self.temperature,
                model=self.groq_model_name
            ).choices[0].message.content

        if verbose:
            for dialog in dialogs[0]:
                print(dialog['role'])
                print(dialog['content'])
                print()
        
        plans = [line.strip() for line in results.split('\n')]
        plans = [line for line in plans if (not line.startswith('#')) and len(line) > 0]

        self.cache[query] = (results, plans)
        return results, plans

    def estimate_height_range(self, target_name, obj_height):
        system_prompt = self.height_range_prompts['system_prompt']
        user1 = self.height_range_prompts['user1']
        assistant1 = self.height_range_prompts['assistant1']

        user2 = f"target: {target_name}\nheight: {obj_height}\ntarget height range: "

        dialogs: List[Dialog] = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user1},
                {"role": "assistant", "content": assistant1},
                {"role": "user", "content": user2}
            ],
        ]

        if self.use_local_llama:
            results = self.generator.chat_completion(
                dialogs,
                max_gen_len=self.max_gen_len,
                temperature=self.temperature,
                top_p=self.top_p,
                logprobs=False,
            )[0]['generation']['content']
        else:
            results = self.generator.chat.completions.create(
                messages=dialogs[0], 
                max_tokens=self.max_gen_len,
                temperature=self.temperature,
                model=self.groq_model_name
            ).choices[0].message.content
        
        results = [line.strip() for line in results.split('\n')]
        results = [line for line in results if (not line.startswith('#')) and len(line) > 0]
        try:
            zrange = np.array(eval(results[0]))
        except:
            zrange = None

        return zrange
