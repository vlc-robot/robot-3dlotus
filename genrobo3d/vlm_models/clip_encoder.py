import os

import torch
import torch.nn as nn

from PIL import Image

from transformers import CLIPModel, AutoTokenizer, CLIPProcessor
import open_clip

def get_prompts_from_label(text):
    template = ['',
        'A ',
        'A model of ',
        'A model of a ',
        'A image of ',
        'A image of a ',
        'A 3D model of ',
        'A 3D model of a ',
        'A rendering model of ',
        'A rendering model of a ',
        'A point cloud of ',
        'A point cloud of a ',
        'A point cloud model of ',
        'A point cloud model of a ',
        'A 3D rendering model of ',
        'A 3D rendering model of a ',
        'A rendering image of ',
        'A rendering image of a ',
        'A 3D rendering image of ',
        'A 3D rendering image of a '
        ]
    template_last = ['.', ' with white background.', ' with black context.']

    prompt_texts = []
    for prefix in template:
        for suffix in template_last:
            prompt_texts.append(prefix + text + suffix)
    
    return prompt_texts


class ClipEncoder(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32', device=None):
        super().__init__()
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.model = CLIPModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)
        self.model.eval()
        
    def forward_text(self, text, use_prompt=True, output_hidden_states=False, **kwargs):
        if use_prompt:
            captions = get_prompts_from_label(text)
        else:
            captions = text
        cap_inputs = self.tokenizer(
            captions, padding=True, truncation=True, return_tensors="pt", max_length=77
        )
        # print(cap_inputs.keys())
        cap_inputs['input_ids'] = cap_inputs['input_ids'].to(self.device)
        cap_inputs['attention_mask'] = cap_inputs['attention_mask'].to(self.device)

        text_outputs = self.model.text_model(
            input_ids=cap_inputs['input_ids'],
            attention_mask=cap_inputs['attention_mask'],
            output_hidden_states=True,
            return_dict=True,
        )
        # TODO: num_tokens are ignored
        if output_hidden_states:
            return text_outputs.last_hidden_state   # (batch, ntokens, hidden_size)
        else:
            pooled_output = text_outputs.pooler_output
            cap_fts = self.model.text_projection(pooled_output)
            # cap_fts = self.model.get_text_features(**cap_inputs)
            return cap_fts
    
    def forward_image(self, images, **kwargs):
        if isinstance(images[0], Image.Image):
            images = self.processor(images=images, return_tensors="pt")['pixel_values']  
        images = images.to(self.device)
        img_fts = self.model.get_image_features(pixel_values=images)
        return img_fts
    
    def forward(self, enc_type, input, **kwargs):
        if enc_type == 'text':
            return self.forward_text(input, **kwargs)
        elif enc_type == 'image':
            return self.forward_image(input, **kwargs)
        else:
            raise NotImplementedError(f'Unknown enc_type: {enc_type}')


class OpenClipEncoder(nn.Module):
    def __init__(self, model_name='ViT-bigG-14', pretrained='laion2b_s39b_b160k', device=None):
        # model_name='ViT-B-32', pretrained='openai'
        super().__init__()
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        # open_clip.list_pretrained()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)
        self.model.eval()

    def forward_text(self, text, use_prompt=True, output_hidden_states=False, **kwargs):
        if use_prompt:
            captions = get_prompts_from_label(text)
        else:
            captions = text
        cap_inputs = self.tokenizer(captions, context_length=77).to(self.device)
        
        if output_hidden_states:
            cast_dtype = self.model.transformer.get_cast_dtype()
            x = self.model.token_embedding(cap_inputs).to(cast_dtype)  # [batch_size, n_ctx, d_model]
            x = x + self.model.positional_embedding.to(cast_dtype)
            # x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.model.transformer(x, attn_mask=self.model.attn_mask)
            # x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection
            # return F.normalize(x, dim=-1) if normalize else x
            num_tokens = cap_inputs.argmax(dim=-1) + 1  # (eot_token is the highest number in each sequence)
            x = [v[:num_tokens[i]] for i, v in enumerate(x)]
            return x
        else:
            cap_fts = self.model.encode_text(cap_inputs)
            return cap_fts
    
    def forward_image(self, images, **kwargs):
        if isinstance(images[0], Image.Image):
            images = torch.stack([self.preprocess(image.convert('RGB')) for image in images], 0)
        images = images.to(self.device)
        img_fts = self.model.encode_image(images)
        return img_fts
    
    def forward(self, enc_type, input, **kwargs):
        if enc_type == 'text':
            return self.forward_text(input, **kwargs)
        elif enc_type == 'image':
            return self.forward_image(input, **kwargs)
        else:
            raise NotImplementedError(f'Unknown enc_type: {enc_type}')

    