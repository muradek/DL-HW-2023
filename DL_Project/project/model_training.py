from roboflow import Roboflow
from ultralytics import YOLO

# load our datasets from roboflow
def load_datasets():
    rf = Roboflow(api_key="BQwCglXoth5FSTCCWMZF")
    # load the train set
    train_project = rf.workspace("technion-fl2u0").project("taco_train_only")
    train_set = train_project.version(1).download("yolov8")
    # load the test set
    test_project = rf.workspace("technion-fl2u0").project("taco_test_set")
    test_set_3 = test_project.version(3).download("yolov8")

    return train_set, test_set_3

# initialize and custom train a YOLOv8 model
def set_model(train_set, model_version):
    model = YOLO(model_version) # initializing a pre-trained YOLOv8 model
    data = str(train_set.location) + "/data.yaml"
    train_results = model.train(data=data, epochs=100, imgsz=640, patience=50, batch=2) # training the model
    return model, train_results

# evaluate the model on our test_set
# saves a predictions.json file for cocoEVAL() 
def evaluate_model(test_set, model):
    data = str(test_set.location) + "/data.yaml"
    val_results = model.val(data=data, save_json=True)# evaluate model performance on the test set
    return val_results

# predict with the model on a test set
def pred(model, dataset):
    source = str(dataset.location) + "/test/images"
    pred_results = model.predict(source=source, save=True, save_json=True)
    return pred_results