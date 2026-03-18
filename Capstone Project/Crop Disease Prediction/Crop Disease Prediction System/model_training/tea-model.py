# ==================== IMPORTS ====================
import torch #pytorch library
import torch.nn as nn #neural network layers
import torch.optim as optim #optimizer
from torchvision import datasets, transforms, models # used to read image data set, data preprocessing and augmentation and pretrained models
from torch.utils.data import DataLoader, random_split #load batch-wise data, split to train and validation set
from sklearn.metrics import confusion_matrix, classification_report #evaluation meterics
import os
import matplotlib.pyplot as plt #for generating plots
import seaborn as sns #for confusion matrix
import pandas as pd

import csv
from datetime import datetime
import numpy as np
# ==================== CONFIG ====================
current=os.path.dirname(os.path.abspath(__file__))
dataset=os.path.abspath(os.path.join(current,'..','data','tea','data'))
img_size=224 #fix input image size for vgg16 224x224
batch_size=32 #32 images are passed to gpu for training
epochs=25 #dataset is passed 15 times for training
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")#use gpu, if not available use cpu
# ==================== TRANSFORMS ====================
train_transform=transforms.Compose([
    transforms.Resize((img_size,img_size)),#resize image to (224x224)
    transforms.RandomHorizontalFlip(),#randomly left-right flips an image
    transforms.RandomRotation(10),# rotate image to upto 10 degrees randomly
    transforms.ColorJitter(brightness=0.2,contrast=0.2),#change brightness and contrast
    transforms.RandomResizedCrop(img_size,scale=(0.9,1.0)), # random crop, atleast 90% remains at all time
    transforms.ToTensor(),# transforms image to tensor(rgb values from 0-255 becomes 0-1)
    transforms.Normalize(
        mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])#normalize image for vgg16,these values are used in pretrained vgg16
])
val_transform=transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
# ==================== DATASET LOAD ====================
transformed_dataset=datasets.ImageFolder(root=dataset,transform=train_transform)#loading dataset
class_names=transformed_dataset.classes#classes to predict from
num_classes=len(class_names)
print("Classes:",class_names)
# ==================== TRAIN/VAL SPLIT (80/20)  ====================
train_size=int(0.8*len(transformed_dataset))#train size=80
val_size=len(transformed_dataset)-train_size#val size=20
train_dataset,val_dataset=random_split(transformed_dataset,[train_size,val_size],generator=torch.Generator().manual_seed(1))#split at fixed seed for reproducable results
val_dataset.dataset.transform=val_transform#transform the splitted dataset
# ==================== LOAD DATA ====================
train_loader=DataLoader(train_dataset,batch_size=batch_size,
    shuffle=True,#randomly shuffles training data each epoch,Prevents model from learning order-based patterns
)
val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False)#shuffle not needed
# ==================== VGG16 BASE MODEL ====================
model=models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)#load pretrained weights
for param in model.features.parameters():
    param.requires_grad=False#freeze its convolutional layers to resuse features like edges,corners,etc

# ==================== CUSTOM CLASSIFIER ====================
model.classifier=nn.Sequential(
    nn.Linear(25088,256),#vgg16 layer give feature map output([512,7,7]),flaten it(512x7x7=25088)and compress to 256 because vgg16 has too many parameters, risking overfittig
    nn.ReLU(),# rectified linear unit (activation function), removes usless noice/features, works well with vgg
    nn.Dropout(0.5),#randomly disable 50% neurons during training to avoid overfitting and learn robus patterns
    nn.Linear(256,num_classes)#predicts on basis of 256 from classes
)
model=model.to(device)
# ==================== LOSS & OPTIMIZER ====================
criterion=nn.CrossEntropyLoss()#loss function,tells how wrong was the prediction
optimizer=optim.Adam(model.classifier.parameters(),lr=1e-4)#updates weights to reduce loss, adam is good at adapting learning rate
# ==================== TRAIN ====================
for epoch in range(epochs):
    model.train()#training starts
    running_loss=0.0# reset counts for each epoch for loss and accuracy
    correct=0
    total=0
    for images, labels in train_loader:
        images,labels=images.to(device),labels.to(device)#send images and labels to gpu
        optimizer.zero_grad()#removes old gradients(changes to weights)
        outputs=model(images)#raw score as output after feeding images to model
        loss=criterion(outputs,labels)#calculates loss, compares prediction with labels
        loss.backward()# gradient is calculated, tells weights how they contribute to error
        optimizer.step()#optimizer updates weights, this is a learning step
        running_loss+=loss.item()#used to calculate average epoch loss
        _,predicted=torch.max(outputs,1)#ignore highest class(_), keep its index(predicted)
        total+=labels.size(0)#for accuracy calculation
        correct+=(predicted==labels).sum().item()#for accuracy calculation
    train_acc=correct/total#accuracy calculation
    print(
        f"Epoch [{epoch+1}/{epochs}] "#print epoch
        f"Loss: {running_loss/len(train_loader):.4f} "#print loss
        f"Accuracy: {train_acc:.4f}"#print accuracy
    )
# ==================== VALIDATION ====================
results=os.path.abspath(os.path.join(current,'..','results','tea'))#to save model's data 
os.makedirs(results,exist_ok=True)# make directory if it doesn't exist
model.eval()#evaluates model, turns on all neurons, dropout off
y_true=[]#true labels
y_pred=[]#predicted labels
with torch.no_grad():#disable gradient calculation,no training, only testing
    for images,labels in val_loader:
        images,labels=images.to(device),labels.to(device)#ensures model,images and labels are on same cpu/gpu
        outputs=model(images)#model predicts images score
        _,predicted=torch.max(outputs,1)#ignore highest class(_), keep its index(predicted)
        y_true.extend(labels.cpu().numpy())#move tensors(container/array) back to cpu
        y_pred.extend(predicted.cpu().numpy())#move tensors(container/array) back to cpu
#==================== SAVE MODEL ====================
save=os.path.abspath(os.path.join(current,'..','saved_models'))
os.makedirs(save,exist_ok=True)
torch.save(model.state_dict(),os.path.join(save,"tea.pth"))#save model
print("\n================ MODEL SAVED SUCCESSFULLY ================")
# ==================== SAVE results ====================
report=classification_report(
    y_true, y_pred,
    target_names=class_names,
    zero_division=0
)
with open(os.path.join(results, "classification_report.txt"),"w") as f:#create classification report as text
    f.write(report)
report_dict=classification_report(
    y_true, y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
    )
df_report=pd.DataFrame(report_dict).transpose()
df_report.to_csv(os.path.join(results,"classification_report.csv"))#save classification report as csv
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=class_names,yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(results,"confusion_matrix.png"))
plt.close()
print("\n================ Classification Report & Confusion Matrix Saved Sucessfully ================")
print("\n================ CONFUSION MATRIX ================\n")
print(cm)#print confusion matrix
print("\n================ CLASSIFICATION REPORT ================\n")
print(report)
# ==================== log ====================
def log_from_dataloader(
        model,
        dataloader,
        class_names,
        output_csv,
        device,
        crop_name,
        dataset_type="validation",
        model_name="vgg16"
    ):
    model.eval()
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "image_id",
                "date",
                "crop",
                "true_label",
                "predicted_label",
                "confidence",
                "is_correct",
                "dataset_type",
                "model_name"
            ])
        img_counter = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, dim=1)
                for i in range(len(labels)):
                    true_idx = labels[i].item()
                    pred_idx = preds[i].item()
                    true_label = class_names[true_idx]
                    pred_label = class_names[pred_idx]
                    writer.writerow([
                        f"{dataset_type}_{img_counter}",
                        datetime.now().strftime("%Y-%m-%d"),
                        crop_name,
                        true_label,
                        pred_label,
                        float(confs[i].item()),
                        int(true_idx == pred_idx),
                        dataset_type,
                        model_name
                    ])
                    img_counter += 1
log_from_dataloader(
    model=model,
    dataloader=val_loader,
    class_names=class_names,
    output_csv=os.path.join(results, "predictions_log.csv"),
    device=device,
    crop_name="tea",
    dataset_type="validation",
    model_name="vgg16"
)