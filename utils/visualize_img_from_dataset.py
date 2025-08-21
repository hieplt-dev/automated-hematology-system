import cv2
import sys
sys.path.append('..')  # Adjust path to import BCCD_DATASET
from bccd_dataset import BCCD_DATASET


if __name__ == "__main__":
	img, annotations = BCCD_DATASET(root='../BCCD_Dataset', mode='train')[0]
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	image = img.copy()

	for anno in annotations:
		boxes = anno['boxes']
		label = anno['labels']
		xmin, ymin, xmax, ymax = boxes
		if label[0] == "R":
			cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
			cv2.putText(image, label, (xmin + 10, ymin + 15),
						cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 255, 0), 1)
		elif label[0] == "W":
			cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
			cv2.putText(image, label, (xmin + 10, ymin + 15),
						cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 0, 255), 1)
		elif label[0] == "P":
			cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
			cv2.putText(image, label, (xmin + 10, ymin + 15),
						cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (255, 0, 0), 1)

	cv2.imshow("test", image)
	cv2.waitKey()
