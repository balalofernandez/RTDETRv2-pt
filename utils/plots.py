import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import torch


def visualize_bboxes(images, gt_boxes, pred_boxes):
    batch_size, channels, height, width = images.shape
    num_imgs = 5
    fig, axs = plt.subplots(2, num_imgs, figsize=(20, 8))
    
    for i in range(num_imgs):
        image = images[i].numpy().transpose(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
        
        # Plot ground truth
        axs[0, i].imshow(image)
        axs[0, i].set_title(f"Ground Truth {i+1}")
        axs[0, i].axis('off')
        
        for box in gt_boxes[i]['boxes'].cpu():
            cx, cy, w, h = box
            x1 = (cx - w / 2) * width
            y1 = (cy - h / 2) * height
            w = w * width
            h = h * height
            
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
            axs[0, i].add_patch(rect)
        
        # Plot predictions
        axs[1, i].imshow(image)
        axs[1, i].set_title(f"Prediction {i+1}")
        axs[1, i].axis('off')
        
        for box in pred_boxes[i]['boxes'].cpu():
            cx, cy, w, h = box
            x1 = (cx - w / 2) * width
            y1 = (cy - h / 2) * height
            w = w * width
            h = h * height
            
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='g', facecolor='none')
            axs[1, i].add_patch(rect)
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.savefig(f'/home/balalo/AIPlayground/RTDETRv2-pt/img.png', dpi=300, bbox_inches='tight')
    fig.canvas.draw()
    img_plot = np.array(fig.canvas.buffer_rgba())
    plt.close(fig)
    return cv2.cvtColor(img_plot, cv2.COLOR_RGBA2RGB)