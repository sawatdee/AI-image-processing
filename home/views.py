import os , sys , re , base64 , json
from django.shortcuts import render_to_response , render
from django.views.generic import TemplateView
from imageai.Detection import ObjectDetection
# from django.utils import simplejson
from datetime import datetime

execution_path = os.getcwd()

def base64ToFilePath(base64Str , basePath = "./home/libs/input/"):
    img_data = bytes(base64Str, 'utf-8')
    fileName = datetime.today().strftime('%Y-%m-%d-%H:%M:%S') + ".png"
    filePath = basePath + fileName
    with open(os.path.join(execution_path ,filePath ), "wb") as fh:
        fh.write(base64.decodebytes(img_data))
    return fileName

def filePathToBase64(filePath):
    base64Str = ''
    with open(os.path.join(execution_path ,filePath ), "rb") as image_file:
        base64Str = base64.b64encode(image_file.read())
    return 'data:image/png;base64,' + base64Str.decode("utf-8")    

prices = {
    'bottle' : 10,
    'apple' : 15,
    'orange' : 15,
    'sandwich' : 20,
    'hot dog' : 25,
    'pizza' : 200,
    'donut' : 40,
    'cake' : 120
}

def processImage(input_file,output_file) :
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(execution_path , "./home/libs/yolo.h5"))
    detector.loadModel()

    custom_objects = detector.CustomObjects(bottle=True, apple=True, orange=True, sandwich=True, hot_dog=True, pizza=True, donut=True, cake=True)
    detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , input_file), output_image_path=os.path.join(execution_path , output_file), minimum_percentage_probability=30)

    totalPrice = 0
    items = []
    for eachObject in detections:
        name = eachObject["name"]
        totalPrice = totalPrice + prices[name]
        items.append([ name , prices[name] ])
        # print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] , " : " , prices[eachObject["name"]] , " Baht" )
    return ( totalPrice  , items )

# result = processImage(sys.argv[1],sys.argv[2])  
# print("result totalPrice",result[0])

class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'index.html', context=None)
    def post(self, request, **kwargs):
        image = request.POST['image']
        image = re.sub('^data:image\/[a-z]+;base64,','', image)
        inputPath = "./home/libs/input/"
        outputPath = "./home/libs/output/"
        fileName = base64ToFilePath(image)
        result = processImage(inputPath + fileName , outputPath + fileName )
        base64Str = filePathToBase64(outputPath + fileName)
        print(json.dumps(result[1], separators=(',', ':')))
        # return render(request, 'index.html', { 'image' : base64Str , 'items' : result[1]})        
        return render(request, 'index.html', { 'image' : base64Str , 'items' : json.dumps(result[1])})        
