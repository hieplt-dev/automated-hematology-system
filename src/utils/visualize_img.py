import cv2
import numpy as np


def visualize_img(img_chw_tensor, boxes_tensor, labels_tensor):
    # img: torch.Tensor(C,H,W) in [0,1]
    img = (img_chw_tensor.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    boxes = boxes_tensor.cpu().numpy()
    labels = labels_tensor.cpu().numpy()

    for (xmin, ymin, xmax, ymax), label in zip(boxes, labels):
        p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        color = (0,255,0) if label==2 else (0,0,255) if label==1 else (255,0,0)
        
        # get class name
        name = 'Unknown'
        if label == 1:
            name = 'WBC'
        elif label == 2:
            name = 'RBC'
        elif label == 3:
            name = 'Platelets'
        cv2.rectangle(image, p1, p2, color, 1)
        cv2.putText(image, name, (p1[0]+6, p1[1]+16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imshow("image", image)
    cv2.waitKey(0)
