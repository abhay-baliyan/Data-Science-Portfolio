import torch
import torch.nn as nn
from torchvision import transforms,models
from PIL import Image
import os
base=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))#root folder
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------- CROP INFO ----------
crop_info={
    "Potato":{
        "model":os.path.join(base,"saved_models","potato.pth"),
        "classes":["Early Blight","Healthy","Late Blight"]
    },
    "Corn":{
        "model":os.path.join(base,"saved_models","corn.pth"),
        "classes":["Common Rust","Grey Leaf Spot","Healthy","Nothern Leaf Blight"]
    },
    "Sugarcane":{
        "model":os.path.join(base,"saved_models","sugarcane.pth"),
        "classes":['Healthy','Mosaic','RedRot','Rust','Yellow']
    },
    "Rice":{
        "model":os.path.join(base,"saved_models","rice.pth"),
        "classes":['Bacterial Leaf Blight','Brown Spot','Healthy','Leaf Blast','Leaf Scald','Narrow Brown Spot']
    },
    "Wheat":{
        "model":os.path.join(base,"saved_models","wheat.pth"),
        "classes":["Aphid","Black Rust","Blast","Brown Rust","Common Root Rot","Fusarium Head Blight","Healthy","Leaf Blight","Mildew","Mite","Septoria","Smut","Stem fly","Tan spot","Yellow Rust"]
    },
    "Cotton":{
        "model":os.path.join(base,"saved_models","cotton.pth"),
        "classes":['Diseased Cotton Leaf','Fresh Cotton Leaf']
    },
    "Tea":{
        "model":os.path.join(base,"saved_models","tea.pth"),
        "classes":["Algal Leaf","Anthracnose","Bird Eye Spot","Brown Blight","Gray Light","Healthy","Red Leaf Spot","White Spot"]
    }
}
# ---------- TRANSFORM ----------
transform=transforms.Compose([
    transforms.Resize((224,224)),#transform image to 224x224 for vgg16
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],#required tensor normalization for vgg16
        std=[0.229,0.224,0.225]
    )
])
# ---------- LOAD MODEL ----------
def load_model(crop):#load model on basis of selection
    ci=crop_info[crop]#selected crop info
    num_classes=len(ci["classes"])#num of current crop classes
    model=models.vgg16(weights=None)#vgg16 architecture to load
    model.classifier=nn.Sequential(
        nn.Linear(25088,256),#vgg16 layer give feature map output([512,7,7]),flaten it(512x7x7=25088)and compress to 256 because vgg16 has too many parameters, risking overfittig
        nn.ReLU(),# rectified linear unit (activation function), removes usless noice/features, works well with vgg
        nn.Dropout(0.5),#randomly disable 50% neurons during training to avoid overfitting and learn robus patterns
        nn.Linear(256,num_classes)#predicts on basis of 256 from classes
    )
    model.load_state_dict(torch.load(ci["model"],map_location=device))#fetch model
    model.to(device)#run using gpu
    model.eval()#evaluate using model
    print(num_classes)
    return model,ci["classes"]
# ---------- PREDICT ----------
def predict_image(image_path,crop):
    model,classes=load_model(crop)
    img=Image.open(image_path).convert("RGB")#open image in rgb
    tensor=transform(img).unsqueeze(0).to(device)
    with torch.no_grad():#gradient calculation freezed
        outputs=model(tensor)#calculate output in form of tensors
        probs=torch.softmax(outputs,dim=1)[0]#convert tensors to probability
    top_probs,top_idxs=torch.topk(probs,k=2)#take top 2 predictions
    top1_idx=top_idxs[0].item()#top prediction
    top2_idx=top_idxs[1].item()#second prediction
    top1_con=top_probs[0].item()*100#conidence conversion
    top2_con=top_probs[1].item()*100#conidence conversion
    return {#return dictionary of required values
        "top1_class":classes[top1_idx],#for results
        "top1_con":top1_con,#for results
        "top2_class":classes[top2_idx],#for results
        "top2_con":top2_con,#for results
        "class_idx":top1_idx,#for grad cam
        "tensor":tensor,#for grad cam
        "model":model,#for grad cam
        "class_names":classes #all the classes
    }