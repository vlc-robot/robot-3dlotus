from PIL import Image
from easydict import EasyDict
import numpy as np

import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection

from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from transformers.image_transforms import center_to_corners_format

MODEL_IDS = {
    "base": "google/owlv2-base-patch16-ensemble",
    "large": "google/owlv2-large-patch14-ensemble",
}

def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.001):
    """
    # Augments
        dets:        boxes coordinate tensor (format:[x1, y1, x2, y2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function. The lower, the more decrease on the weight
        thresh:      score thresh
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = box_scores.clone()
    # areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    areas = (x2 - x1) * (y2 - y1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())
        
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].int()

    return keep

class Owlv2ObjectDetector(object):
    def __init__(self, model_id, device=None) -> None:
        """
        Load base/large model for open-vocabulary object detection
        """
        if model_id == 'base':
            model_id = MODEL_IDS[model_id]
        elif model_id == 'large':
            model_id = MODEL_IDS[model_id]
        else:
            raise NotImplementedError(f'model_id should be in {list(MODEL_IDS.keys())}.')
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id, device_map=self.device)

        self.image_size = self.processor.image_processor.size
        self.image_size = [self.image_size['width'], self.image_size['height']]

    @torch.no_grad()
    def encode_images(self, images):
        '''Args:
        - images: ndarray, (batch_size, height, width, 3)
        '''
        images = [Image.fromarray(image) for image in images]

        # {'pixel_values': (batch_size, nchannels, height, width)}
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        vision_outputs = self.model.owlv2.vision_model(
            pixel_values=inputs['pixel_values'],
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        # Get image embeddings, shape=(batch_size, 1+num_patches*num_patches=3601, hidden_size=768)
        image_embeds = self.model.owlv2.vision_model.post_layernorm(
            vision_outputs.last_hidden_state
        )

        # Resize class token
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], image_embeds[:, :-1].shape)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.model.layer_norm(image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        batch_size = image_embeds.shape[0]
        num_patches = self.model.sqrt_num_patches
        hidden_dim = image_embeds.shape[-1]
        feature_map = image_embeds.reshape((batch_size, num_patches, num_patches, hidden_dim))

        # Predict object classes [batch_size, num_patches, num_queries+1]
        image_class_embeds = self.model.class_head.dense0(image_embeds)
        
        # # Normalize image features
        # image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)

        # Apply a learnable shift and scale to logits
        class_logit_shift = self.model.class_head.logit_shift(image_embeds)
        class_logit_scale = self.model.class_head.logit_scale(image_embeds)
        class_logit_scale = self.model.class_head.elu(class_logit_scale) + 1

        # Predict objectness
        objectness_logits = self.model.objectness_predictor(image_embeds)

        # Predict object boxes
        pred_boxes = self.model.box_predictor(image_embeds, feature_map)

        return EasyDict(
            image_embeds=feature_map.data.cpu(),
            pred_boxes=pred_boxes.data.cpu(),
            objectness_logits=objectness_logits.data.cpu(),
            image_class_embeds=image_class_embeds.data.cpu(),
            class_logit_shift=class_logit_shift.data.cpu(),
            class_logit_scale=class_logit_scale.data.cpu(),
        )

    @torch.no_grad()
    def encode_texts(self, texts):
        '''Args:
        - texts: a list of strings
        '''
        # {'input_ids': (n_txts, max_txt_len), 'attention_mask': (n_txts, max_txt_len)}, max_txt_len=16
        inputs = self.processor(text=texts, return_tensors="pt").to(self.device)

        # Get embeddings for all text queries in all batch samples
        text_outputs = self.model.owlv2.text_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        text_embeds = text_outputs.pooler_output
        text_embeds = self.model.owlv2.text_projection(text_embeds)

        # # normalized features
        # text_embeds = text_embeds / torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True)

        return EasyDict(text_embeds=text_embeds.data.cpu())

    def predict_class_logits(self, image_outputs, text_outputs):
        # Normalize image and text features
        image_class_embeds = image_outputs.image_class_embeds   # (n_imgs, num_patches**2, dim)
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        query_embeds = text_outputs.text_embeds                 # (n_txts, dim)
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)

        # Get class predictions
        pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)

        # Apply a learnable shift and scale to logits
        pred_logits = (pred_logits + image_outputs.class_logit_shift) * image_outputs.class_logit_scale

        return pred_logits

    def get_unnormalized_images(self, images):
        images = [Image.fromarray(image) for image in images]

        # {'pixel_values': (batch_size, nchannels, height, width)}, height=width
        inputs = self.processor(images=images, return_tensors="np")
        pixel_values = inputs['pixel_values']

        unnormalized_images = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        unnormalized_images = (unnormalized_images * 255).astype(np.uint8)
        unnormalized_images = np.moveaxis(unnormalized_images, 1, -1)
        unnormalized_images = [Image.fromarray(x) for x in unnormalized_images]

        return unnormalized_images

    def post_process_objectness_detection(
            self, image_outputs, threshold=0.1, target_sizes=None, 
            min_size_ratio=None, max_size_ratio=0.8, 
            min_return_topk=None, max_return_topk=None,
            use_nms=False, nms_sigma=0.2, nms_thresh=0.1,
        ):
        """
        Converts the raw output into final bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.
        Remove too small or too large bboxes with size < min_size_ratio and size > max_size_ratio.
        Returns at most max_return_topk boxes with score >= threshold. 
        If all scores < threshold, return min_return_topk boxes.
        
        Returns:
            results: list of {"scores": (num_bboxes, ), "boxes":, "path_indexs":, "patch_coords":}
                     len(results) == batch_size
        """
        # (batch_size, num_patches**2)
        objectness_scores = torch.sigmoid(image_outputs.objectness_logits).cpu()

        # Convert to [x0, y0, x1, y1] format
        box_sizes = torch.prod(image_outputs.pred_boxes[..., 2:], -1).cpu()
        boxes = center_to_corners_format(image_outputs.pred_boxes).cpu()

        results = []
        for s, b, bsize in zip(objectness_scores, boxes, box_sizes):
            obj_ids = torch.arange(s.size(0))
            if min_size_ratio is not None:
                obj_ids = obj_ids[bsize[obj_ids] > min_size_ratio]
            if max_size_ratio is not None:
                obj_ids = obj_ids[bsize[obj_ids] < max_size_ratio]

            tmp_obj_ids = obj_ids[s[obj_ids] >= threshold]
            if len(tmp_obj_ids) == 0 and min_return_topk is not None:
                obj_ids = obj_ids[torch.topk(s[obj_ids], min_return_topk).indices]
            else:
                obj_ids = tmp_obj_ids
            
            obj_ids = obj_ids[s[obj_ids].argsort(descending=True)]
            if max_return_topk is not None:
                obj_ids = obj_ids[:max_return_topk]

            score = s[obj_ids]
            box = b[obj_ids]
            patch_index = torch.arange(s.size(0), dtype=torch.long)[obj_ids]
            patch_coord = torch.stack([patch_index % self.model.sqrt_num_patches, patch_index // self.model.sqrt_num_patches], -1) / self.model.sqrt_num_patches

            if target_sizes is not None:
                # the image is padded: https://github.com/huggingface/transformers/issues/27205
                img_w, img_h = target_sizes
                img_size = max(img_w, img_h)
                scale_fct = torch.FloatTensor([img_size,img_size, img_size, img_size])
                box = box * scale_fct[None, :]
                patch_coord = patch_coord * scale_fct[None, :2]

            if use_nms:
                obj_ids = soft_nms_pytorch(box, score, sigma=nms_sigma, thresh=nms_thresh)
                score = score[obj_ids]
                box = box[obj_ids]
                patch_index = patch_index[obj_ids]
                patch_coord = patch_coord[obj_ids]

            results.append({"scores": score, "boxes": box, "patch_indexs": patch_index, "patch_coords": patch_coord})
        
        return results



    
    