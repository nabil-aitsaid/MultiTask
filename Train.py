
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from MultiTaskCNN import MultiTaskCNN
from MultiTaskLoss import MultiTaskLoss
import torch.nn as nn
import  torch.optim
from visualization import vis

class ExtractFirstChannel(object):
    def __call__(self, img):
        return img[0, :, :].reshape(1,64,64)
    
batchsize = 128
transform = transforms.Compose([
    transforms.ToTensor(),           # Convert the image to a PyTorch tensor
    ExtractFirstChannel(),     # the image is in black & white so we extract only first channel
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the image
])
traindataset = datasets.ImageFolder(root='/home/nabil/masteriales/datasets/multimnist/train', transform=transform)
trainloader = DataLoader(traindataset,batch_size=batchsize,shuffle=True)

testdataset = datasets.ImageFolder(root='/home/nabil/masteriales/datasets/multimnist/test', transform=transform)
testloader = DataLoader(testdataset,batch_size=batchsize)

number_of_batches = len(trainloader)

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")

# initialize the model
model = MultiTaskCNN()
model=model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn= MultiTaskLoss()
print(" --- training step ---")
model.train()
epochs = 20
losses_history = [ ]
for epoch in range(epochs) :
    run_loss=0

    for inputs,labels in trainloader:
        
        # split the labels ex: "90" => [9,0]
        labels = torch.tensor([ [int(traindataset.classes[e][0]) , int(traindataset.classes[e][1]) ] for e in labels ])
        inputs = inputs.to(device)
        labels = labels.to(device)

        # make predictions
        outputs=model(inputs)
        #initialize gradients
        optimizer.zero_grad()
        #compute loss
        loss = loss_fn(outputs,labels)
        #backpropagation
        loss.backward()
        #update weights
        optimizer.step()
        run_loss+=loss.item()
    
    run_loss = run_loss / number_of_batches
    print(f"Epoch: [{epoch + 1}/{epochs}], Loss: {run_loss}")
    losses_history.append(run_loss)
    vis(losses_history)


# evaluation step
print(" --- evaluation step ---")
print(" --- computing training score ---")
number_of_batches = len(trainloader)
model.eval()
correct_predictions=0
total=0
for inputs, labels in trainloader:
    labels = torch.tensor([ [int(traindataset.classes[e][0]) , int(traindataset.classes[e][1]) ] for e in labels ])
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)

    for i in range(len(outputs)):
        total+=1
        if( (outputs[i,0,:].argmax()==labels[i,0]) and (outputs[i,1,:].argmax()==labels[i,1]) ):
            correct_predictions+=1
accuracy = correct_predictions/total
print(f'Train Accuracy: {accuracy * 100:.2f}%')

# evaluation step
print(" --- computing test score ---")
number_of_batches = len(testloader)
model.eval()
correct_predictions=0
total=0
for inputs, labels in testloader:
    labels = torch.tensor([ [int(testdataset.classes[e][0]) , int(testdataset.classes[e][1]) ] for e in labels ])
    inputs = inputs.to(device)
    labels = labels.to(device)    
    outputs = model(inputs)

    for i in range(len(outputs)):
        total+=1
        if( (outputs[i,0,:].argmax()==labels[i,0]) and (outputs[i,1,:].argmax()==labels[i,1]) ):
            correct_predictions+=1
accuracy = correct_predictions/total
print(f'Test Accuracy: {accuracy * 100:.2f}%')
