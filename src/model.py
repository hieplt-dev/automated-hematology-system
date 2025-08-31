from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class BCCD_Model:
    def __init__(self, num_classes):
        # Load a pre-trained Faster R-CNN model with a ResNet-50 backbone
        self.model = fasterrcnn_resnet50_fpn()
        
        # Replace the classifier with a new one for our specific number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        num_classes = num_classes + 1  # +1 for background class
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)