from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ground_truth_path = "annotations_test.json"
output_path = "../runs/detect/val/predictions.json"

#load annotation files
cocoGt=COCO(ground_truth_path)
cocoDt=cocoGt.loadRes(output_path)

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt, 'bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()