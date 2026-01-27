import cv2
import numpy as np


def visualize_img(
    img_chw_tensor, boxes_tensor, labels_tensor, show=True, win_name="image"
):
    """
    - img_chw_tensor: torch.Tensor shape (C,H,W), pixel [0,1]
    - boxes_tensor:   torch.Tensor shape (N,4) with (xmin,ymin,xmax,ymax)
    - labels_tensor:  torch.Tensor shape (N,) with int labels
    """
    # to HWC uint8 BGR
    img = (img_chw_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(
        np.uint8
    )
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    boxes = boxes_tensor.detach().cpu().numpy()
    labels = labels_tensor.detach().cpu().numpy().astype(int)

    # label ↔ name & color
    id2name = {1: "RBC", 2: "WBC", 3: "Platelets"}
    id2color = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}  # BGR

    # draw boxes
    for (xmin, ymin, xmax, ymax), label in zip(boxes, labels):
        p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        color = id2color.get(label, (200, 200, 200))
        name = id2name.get(label, "Unknown")
        cv2.rectangle(image, p1, p2, color, 1)
        cv2.putText(
            image,
            name,
            (p1[0] + 6, p1[1] + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    # count per class
    counts = {k: 0 for k in id2name.keys()}
    for label in labels_tensor.numpy():
        counts[label] += 1

    # Order lines the way you like
    lines = [
        f"RBC: {counts.get(1,0)}",
        f"WBC: {counts.get(2,0)}",
        f"Platelets: {counts.get(3,0)}",
    ]

    # top-right summary panel
    H, W = image.shape[:2]
    pad = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # compute box size from max text width/height
    text_sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
    line_h = max(h for (w, h) in text_sizes)
    text_w = max(w for (w, h) in text_sizes)
    box_w = text_w + 2 * pad
    box_h = line_h * len(lines) + 2 * pad + (len(lines) - 1) * 4  # 4px between lines

    # anchor at top-right (with 10px margin)
    margin = 10
    x2, y1 = W - margin, margin
    x1, y2 = x2 - box_w, y1 + box_h + 8

    # draw semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)  # black
    alpha = 0.35
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # put texts (white)
    y = y1 + pad + line_h
    for t in lines:
        cv2.putText(
            image,
            t,
            (x1 + pad, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_h + 7

    if show:
        cv2.imshow(win_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # also return for saving/logging
    # return image, {"RBC": counts.get(1,0), "WBC": counts.get(2,0), "Platelets": counts.get(3,0)}
    # also return for saving/logging
    # return image, {"RBC": counts.get(1,0), "WBC": counts.get(2,0), "Platelets": counts.get(3,0)}


def visualize_prediction_data(image_np, boxes, labels, scores=None):
    """
    Visualizes predictions on an image (numpy array, BGR or RGB).
    
    Args:
        image_np: numpy array of shape (H, W, 3). Assumed BGR if coming from cv2, or user handles conversion.
                  The function does drawing in BGR (opencv default).
        boxes: List of [xmin, ymin, xmax, ymax]
        labels: List of int labels
        scores: List of float scores (optional)
        
    Returns:
        image_drawn: numpy array with annotations
    """
    image = image_np.copy()
    h_img, w_img = image.shape[:2]
    
    # label ↔ name & color
    id2name = {1: "RBC", 2: "WBC", 3: "Platelets"}
    id2color = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}  # BGR

    # draw boxes
    for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        xmin = xmin/480*w_img
        ymin = ymin/480*h_img
        xmax = xmax/480*w_img
        ymax = ymax/480*h_img
        label = int(labels[i])
        p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        color = id2color.get(label, (200, 200, 200))
        name = id2name.get(label, "Unknown")
        
        cv2.rectangle(image, p1, p2, color, 1) # Thin line
        
        # Determine text
        text = name
        if scores is not None:
            text += f" {int(scores[i]*100)}%"
            
        cv2.putText(
            image,
            text,
            (p1[0] + 6, p1[1] + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    # count per class
    counts = {k: 0 for k in id2name.keys()}
    for label in labels:
        counts[int(label)] += 1

    # Order lines
    lines = [
        f"RBC: {counts.get(1,0)}",
        f"WBC: {counts.get(2,0)}",
        f"Platelets: {counts.get(3,0)}",
    ]

    # top-right summary panel
    H, W = image.shape[:2]
    pad = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    text_sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
    line_h = max(h for (w, h) in text_sizes)
    text_w = max(w for (w, h) in text_sizes)
    box_w = text_w + 2 * pad
    box_h = line_h * len(lines) + 2 * pad + (len(lines) - 1) * 4

    # anchor at top-right (with 10px margin)
    margin = 10
    x2, y1 = W - margin, margin
    x1, y2 = x2 - box_w, y1 + box_h + 8

    # draw semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)  # black
    alpha = 0.35
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # put texts (white)
    y = y1 + pad + line_h
    for t in lines:
        cv2.putText(
            image,
            t,
            (x1 + pad, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_h + 7

    return image
