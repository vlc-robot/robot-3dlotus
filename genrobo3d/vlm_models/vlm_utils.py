import os
import numpy as np



def get_color_palette():
    def hex_to_rgb(hex):
        rgb = []
        for i in (0, 2, 4):
            decimal = int(hex[i:i+2], 16)
            rgb.append(decimal)
        return list(rgb)
    
    import matplotlib.colors as mcolors
    palette = []
    for k, v in mcolors.TABLEAU_COLORS.items():
        palette.append(hex_to_rgb(v[1:]))
    return palette

def draw_contour(rgb_img, mask, color):
    """
    Args:
        rgb_img: ndarray (height, width, 3)
        mask: ndarray (height, width)
        color: (r, g, b)
    Returns:
        rgb_img: rgb_img with colored contour
    """
    import cv2

    # # shift left
    # shift_mask = np.zeros(mask.shape, dtype=mask.dtype)
    # shift_mask[:, :-1] = mask[:, 1:]
    # v_contour = shift_mask ^ mask
    # # shift up
    # shift_mask = np.zeros(mask.shape, dtype=mask.dtype)
    # shift_mask[:-1] = mask[1:]
    # h_contour = shift_mask ^ mask
    # # merge
    # contour = v_contour | h_contour
    # rgb_img[contour] = color

    contour, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb_img, contour, -1, color, 1)

    return rgb_img


def weighted_average_embeds(embeds, scores, keepdim=False):
    """
    Args:
        embeds: torch.Tensor (batch_size, dim)
        scores: torch.Tensor, (batch_size, )
    """
    normed_scores = scores / scores.sum()

    avg_embeds = (embeds * normed_scores.unsqueeze(1)).sum(dim=0, keepdim=keepdim)
    return avg_embeds
