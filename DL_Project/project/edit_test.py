import os
from pycocotools.coco import COCO

# editing test_set s.t model.val(test_set) will produce
# annotation.json file that fits cocoEval() comparison

annotations_path = os.getcwd() + "/project/annotations_test.json"
print(annotations_path)
coco = COCO(annotations_path)
img_ids = coco.getImgIds()

# find the id of an img based on its name
def get_image_id(img_name):
    id = -1
    for img_id in img_ids:
        img = coco.loadImgs(img_id)[0]
        if img['file_name'] == img_name:
            id = img_id
    return id

def edit_images_name():
    img_dir = os.getcwd() + "/TACO_test_set-3/valid/images"
    for full_img_name in os.listdir(img_dir):
        file_name = (full_img_name.split("_jpg")[0]) + ".jpg" # the name that appears in the json file
        id = get_image_id(file_name)
        if id == -1:
            file_name = (full_img_name.split("_JPG")[0]) + ".JPG" # the name that appears in the json file
            id = get_image_id(file_name)
        
        old_path = img_dir + "/" + full_img_name
        new_path = img_dir + "/" + str(id) + ".jpg"
        os.rename(old_path, new_path)

def edit_labels_name():
    lbl_dir = os.getcwd() + "/TACO_test_set-3/valid/labels"
    for full_lbl_name in os.listdir(lbl_dir):
        file_name = (full_lbl_name.split("_jpg")[0]) + ".jpg" # the name that appears in the json file
        id = get_image_id(file_name)
        if id == -1:
            file_name = (full_lbl_name.split("_JPG")[0]) + ".JPG" # the name that appears in the json file
            id = get_image_id(file_name)
        
        old_path = lbl_dir + "/" + full_lbl_name
        new_path = lbl_dir + "/" + str(id) + ".txt"
        os.rename(old_path, new_path)

edit_images_name()
edit_labels_name()