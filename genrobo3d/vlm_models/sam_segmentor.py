import numpy as np
import matplotlib.pyplot as plt

import torch

from transformers import SamModel, SamProcessor

MODEL_IDS = {
   "base": "facebook/sam-vit-base",
   "huge": "facebook/sam-vit-huge",
}
class SAMSegmentor(object):
    def __init__(self, model_id, device=None):
        if model_id == 'base':
            model_id = MODEL_IDS[model_id]
        elif model_id == 'huge':
            model_id = MODEL_IDS[model_id]
        else:
            raise NotImplementedError(f'model_id should be in {list(MODEL_IDS.keys())}.')
        
        if device is None:
           self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
           self.device = device

        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id, device_map=self.device)

        self.image_longest_edge = self.processor.image_processor.size['longest_edge']
        
    @torch.no_grad()
    def __call__(self, images, boxes, points=None, keep_best_mask=True):
        """
        Args:
            images: ndarray, (batch_size, height, width, 3)
            boxes: list of shape=(num_boxes, 4)
            points: list of shape=(num_boxes, 2)
        """
        # {"pixel_values": (b, 3, h, w), "original_sizes": (b, 2), "reshaped_input_sizes": (b, 2)}
        # padded, reshape_size=(image_longest_edge, image_longest_edge)
        inputs = self.processor(images, return_tensors='pt').to(self.device)

        # (batch_size, 256, 64, 64)
        image_embeddings = self.model.get_image_embeddings(inputs['pixel_values'])

        num_images = len(images)

        results = []
        for i in range(num_images):
            if len(boxes[i]) == 0:
               results.append(None)
            else:
                i_boxes = [boxes[i]]
                i_points = [points[i]] if points is not None else None
                i_inputs = self.processor(
                    images[i], input_boxes=i_boxes, input_points=i_points, return_tensors="pt"
                ).to(self.device)
                # print(i_inputs.keys())
                # print(i_inputs["original_sizes"], i_inputs["reshaped_input_sizes"])
                # print(i_inputs['input_boxes'])

                if 'input_points' in i_inputs:
                    i_inputs["input_points"] = i_inputs["input_points"].transpose(1, 2)
                i_inputs.pop("pixel_values", None)
                i_inputs.update({"image_embeddings": image_embeddings[i].unsqueeze(0)})

                # iou_scores: (1, num_boxes, 3), pred_masks: (1, num_boxes, 3, 256, 256)
                i_outputs = self.model(**i_inputs)
                # print(i_outputs.pred_masks.shape)

                # (num_boxes, 3, 256, 256)
                i_masks = self.processor.image_processor.post_process_masks(
                    i_outputs.pred_masks.cpu(), i_inputs["original_sizes"].cpu(), 
                    i_inputs["reshaped_input_sizes"].cpu()
                )[0]
                # print(i_masks.shape)
                i_scores = i_outputs.iou_scores.cpu()[0]

                if keep_best_mask:
                   best_mask_id = i_scores.argmax(dim=1)
                   i_scores = i_scores.gather(1, best_mask_id[:, None])
                   img_h, img_w = i_masks.size()[2:]
                   best_mask_id = best_mask_id[:, None, None, None].expand(-1, -1, img_h, img_w)
                   i_masks = i_masks.gather(1, best_mask_id)

                results.append({"scores": i_scores, "masks": i_masks})
        
        return results



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
        # color = np.array([30/255, 0/255, 0/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_masks_on_image(raw_image, masks, scores):
    print(masks.shape, scores.shape)
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if len(scores.shape) > 1 and scores.shape[0] == 1:
      scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
      mask = mask.cpu().detach()
      if nb_predictions > 1:
        axes[i].imshow(np.array(raw_image))
        show_mask(mask, axes[i])
        axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
        axes[i].axis("off")
    else:
        axes.imshow(np.array(raw_image))
        show_mask(mask, axes)
        axes.title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
        axes.axis("off")
    plt.show()