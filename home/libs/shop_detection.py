from imageai.Detection import ObjectDetection
import os
import sys

prices = {
    'bottle' : 11,
    'apple' : 20,
    'orange' : 20,
    'sandwich' : 20,
    'hot_dog' : 20,
    'pizza' : 20,
    'donut' : 20,
    'cake' : 20
}

def processImage(input_file,output_file) :
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    execution_path = os.getcwd()
    # input_file = sys.argv[1]
    # output_file = sys.argv[2]


    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
    detector.loadModel()

    custom_objects = detector.CustomObjects(bottle=True, apple=True, orange=True, sandwich=True, hot_dog=True, pizza=True, donut=True, cake=True)
    detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , input_file), output_image_path=os.path.join(execution_path , output_file), minimum_percentage_probability=30)

    totalPrice = 0
    for eachObject in detections:
        totalPrice = totalPrice + prices[eachObject["name"]]
        # print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] , " : " , prices[eachObject["name"]] , " Baht" )
        # print("================================")
    return ( totalPrice , output_file )

result = processImage(sys.argv[1],sys.argv[2])  
print("result totalPrice",result[0])
