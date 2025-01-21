import os

# change directory structure and images names before uploading to roboflow

# --1-- all images are added to the same directory with unique names
# annotaion files were adjusted to the new naming
directory = 'data'
for batch in os.listdir(directory):
    for img_name in os.listdir('data/' + batch):
        old_name = 'data/' + batch + "/" + img_name
        new_name = "new_imgs/" + batch + "_" + img_name
        os.rename(old_name, new_name)


# --2-- splitting the images to train and test folders
def in_test_file(img_name): 
    with open('annotations_test.json') as test_file: 
         return (img_name in test_file.read())


for img_name in os.listdir("new_imgs"):
    if in_test_file(img_name) :
        prev_path = "new_imgs/" + img_name
        new_path = "test_imgs/" + img_name
        os.rename(prev_path, new_path)