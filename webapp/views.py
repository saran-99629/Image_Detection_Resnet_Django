from django.shortcuts import render

# deeplearning libray
from keras.applications.resnet import ResNet50
from keras.preprocessing import image 
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib as plt  
from skimage.io import imread 


from .forms import ImageUploadForm
# file saving funtion
def handle_uploaded_file(f):
    with open('img.jpg', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def index(request):
    return render(request,'webapp/index.html')





def imageprocess(request):
    form = ImageUploadForm(request.POST, request.FILES)
    if form.is_valid():
        handle_uploaded_file(request.FILES['image'])
        model=ResNet50(weights='imagenet')
        img_path= 'img.jpg'

        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        print('predicted: ', decode_predictions(preds, top=3)[0])

        html = decode_predictions(preds, top=3)[0]
        res=[]
        for e in html:
            res.append((e[1],np.round(e[2]*100,2)))
        return render(request,'webapp/result.html',{'res':res})

    return render(request,'webapp/index.html')