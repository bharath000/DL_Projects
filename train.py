from dataloaders.aptos import AptosDataset
from CustomTransforms.customtransform import Rescale, RandomShear, RandomShift, RandomRotate, RandomScaling, Sharpen, Brightness, Contrast, ToTensor
from torchvision import transforms, utils
from utility.grid_plot import plot_images 
from scipy import ndimage
import numpy as np 
import torch.nn.functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler
from models.s4nd import DenseNet
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import time
from sklearn.metrics import cohen_kappa_score, accuracy_score

class FlattenedLoss():
    "Same as `func`, but flattens input and target."
    def __init__(self, func, *args, axis:int=-1, floatify:bool=False, is_2d:bool=True, **kwargs):
        self.func,self.axis,self.floatify,self.is_2d = func(*args,**kwargs),axis,floatify,is_2d
        functools.update_wrapper(self, self.func)

    def __repr__(self): return f"FlattenedLoss of {self.func}"
    @property
    def reduction(self): return self.func.reduction
    @reduction.setter
    def reduction(self, v): self.func.reduction = v

    def __call__(self, input, target, **kwargs):
        input = input.transpose(self.axis,-1).contiguous()
        target = target.transpose(self.axis,-1).contiguous()
        if self.floatify: target = target.float()
            
        # Label smoothing experiment
        target = (target * 0.9 + 0.05)
        target[:,0] = 1

        input = input.view(-1,input.shape[-1]) if self.is_2d else input.view(-1)
        return self.func.__call__(input, target.view(-1), **kwargs)
def LabelSmoothBCEWithLogitsFlat(*args, axis:int=-1, floatify:bool=True, **kwargs):
    "Same as `nn.BCEWithLogitsLoss`, but flattens input and target."
    return FlattenedLoss(nn.BCEWithLogitsLoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)

def validate(model, criterion, testing_set):
    accuracy = 0
    test_loss = 0
    model.eval() # Evaluation mode
    with torch.no_grad():
        for images, labels in testing_set:
            
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device)

            output = model.forward(images)
            accuracy += (torch.max(output, 1)[1].view(labels.size()) == label).sum().item()
            test_loss += criterion(output, labels).item()
            print(test_loss)
            print(accuracy)


            # Take exponential to get log softmax probibilities
            #probs = torch.exp(output)
            #print(probs)

            

            # highest probability is the predicted class
            # compare with true label
            #correct_predictions = (labels == probs.max(1)[1])

            # Turn ByteTensor into np_array to calculate mean
            #accuracy += (correct_predictions).mean()
    
    model.train() # Switch training mode back on

def ordinal_accuracy(output,ground_truths, th):
    mask = output >= th
    label = torch.sum(mask, 1) - 1
    #print(label)
    gt = torch.sum(ground_truths, 1) - 1
    #print(gt)
    #print(label,gt)
    return (label == gt).sum().item()

    
    return test_loss/len(testing_set), accuracy/len(testing_set)

def test_model(model_path,device, criterion, val_generator):
    accuracy = 0
    test_loss = 0
    model = DenseNet()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval() # Evaluation mode
    with torch.no_grad():
        for images, labels in val_generator:
            
            
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device)

            output = model(images)
            print(output)
            print(labels)
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #accuracy += pred.eq(labels.view_as(pred)).sum().item()
            #accuracy += ordinal_accuracy(output,labels,0.5)
            accuracy += (torch.max(output, 1)[1].view(labels.size()) == labels).sum().item()
            #test_loss += criterion(output, labels).item()
            #print(test_loss)
            #print(accuracy)
            #e = time.time()

            
    
    
    val_loss,accuracy = test_loss/len(val_generator.dataset), accuracy/len(val_generator.dataset)

    

    #val_loss,accuracy = validate(model, criterion, val_generator) 
    print("Epoch: {} of {}, ".format(epoch+1, num_epochs),
            
            "Test Loss: {:.3f}, ".format(val_loss),
            "Val_Accuracy: %{:.1f}".format(accuracy*100))
    
   


""""
#### code for saving the model
if dev_acc > best_dev_acc:

                # found a model with better validation set accuracy

                best_dev_acc = dev_acc
                snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc, dev_loss.item(), iterations)

                # save model, delete previous 'best_snapshot' files
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

"""""






#plot_images(sample)
#print(sample.shape, y)


if __name__ == '__main__':


    ### Trian and Validation Split of data #####
    df = pd.read_csv('D:/MSCS/ADL/Data/train.csv')
    msk = np.random.rand(len(df)) < 0.8
    train_df = df[msk]
    val_df = df[~msk]

   
    (unique, counts) = np.unique([i for i in train_df.iloc[:,1]], return_counts=True)

    #class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    samples_weight = 1./counts
    #print(samples_weight)
    samples_weight = np.array([samples_weight[i] for i in train_df.iloc[:,1]])
    #print(samples_weight)
    samples_weight = torch.from_numpy(samples_weight)
    #samples_weigth = samples_weight.double()
    weighted_sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=False)
    #exit()

    #df_test = pd.read_csv('D:/MSCS/ADL/Data/test.csv')
    # print(len(val_df[val_df["diagnosis"] == 0]))
    # print(len(train_df[train_df["diagnosis"] == 0]))
    # exit()

    ##### Spliting Trainning Data into Train and Validation data #####

    ##### LOADING TRAIN DATA ######

    aptos_dataset_train = AptosDataset(train_df, 'D:/MSCS/ADL/Data/train_images/',
                                    transform=transforms.Compose([
                                                Brightness(20),
                                                
                                                #Contrast(0.2),
                                                Rescale((224,224)),
                                                Sharpen(0.5),
                                                #Sharpen(2), 
                                                RandomShear(0.1, p=.2),
                                                #RandomShift((10, -20)),
                                                RandomRotate(180.0,p=0.2),
                                                RandomRotate(-180.0,p=0.2),
                                                #RandomScaling((1, 1), p=0.5),
                                                #RandomCrop(224),
                                                ToTensor(),
                                                #transforms.ToTensor(),
                                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                               
                                                #ToNormalize()
                                            ]))
    aptos_dataset_val = AptosDataset(val_df, 'D:/MSCS/ADL/Data/train_images/',
                                    transform=transforms.Compose([
                                                Brightness(20),
                                                #Contrast(0.2),
                                                Sharpen(0.5),
                                                Rescale((224,224)),
                                                ToTensor(),
                                                # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                            ])) 
    aptos_dataset_test = AptosDataset('D:/MSCS/ADL/Data/test.csv', 'D:/MSCS/ADL/Data/test_images/',
                                    transform=transforms.Compose([
                                                Brightness(20),
                                                #Contrast(0.2),
                                                Rescale((256,256)),
                                                ToTensor(),
                                                # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                            ]))                                           


    train_generator = DataLoader(aptos_dataset_train, batch_size=16,
                             num_workers=0,sampler=weighted_sampler)
    val_generator = DataLoader(aptos_dataset_val, batch_size=16,
                            shuffle=False, num_workers=1)
    test_generator = DataLoader(aptos_dataset_test, batch_size=4,
                            shuffle=False, num_workers=1)




    ##### LOADING Validation Data #####


    #sample, y  = aptos_dataset[0]





    # for i_batch, sample_batched in enumerate(train_generator):
    #     x_train, label = sample_batched
    #     print(x_train.shape, label.shape)
    #     #exit()
    #     plot_images(sample_batched)
        # if i_batch == 1:
            
        #     plot_images(sample_batched)
            
        #     break



    # CUDA for PyTorch
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # criterion = nn.CrossEntropyLoss().to(device)
    # print("Finding Accuracy on test data.....")
    # test_model("D:\MSCS\ADL\DL_Projects\saved_models\model_floss_apts_densenet_ot_0.pt",device,criterion,train_generator)
    # exit()

    model  = DenseNet()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    #optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # focal_loss = torch.hub.load(
	# 'adeelh/pytorch-multi-class-focal-loss',
	# model='focal_loss',
	# alpha=[0.00068918, 0.00344828, 0.00125786, 0.00632911, 0.00416667],
	# gamma=2,
	# reduction='mean',
	# device=device,
	# dtype=torch.float32,
	# force_reload=False
    # )   
    #criterion = nn.CrossEntropyLoss().to(device)
    criterion =  nn.BCEWithLogitsLoss().to(device)
    #torch.tensor([0.0005,0.002,0.001,0.005,0.003]).to(device)

    init_loss = 0
    num_batch_print = 250
    num_epochs = 10
    val_print = 1
    for epoch in range(num_epochs):
        model.train()
        n_correct, n_total = 0, 0
        best_val_acc = 0
        for i_batch, sample_batched in enumerate(train_generator):
            
            val_print += 1
            x_train, label = sample_batched
            #print(label.shape, x_train.shape)
            
            x_train, label = x_train.to(device, dtype=torch.float), label.to(device)
            #optimzer 
            optimizer.zero_grad()
            #input into model
            output = model(x_train)
            #print(output)
            #n_correct += (torch.max(output, 1)[1].view(label.size()) == label).sum().item()
            #_, predicted = torch.max(output.data, 1)

            #n_correct += len((torch.max(output, 1)[1] == torch.max(label, 1)[1]).nonzero())
            n_correct += ordinal_accuracy(output, label, 0.5)
            n_total += x_train.shape[0]
            train_acc = 100. * n_correct/n_total
            # output = output.transpose(-1,-1).contiguous()
            # label = label.transpose(-1,-1).contiguous()
            # label = label.float()
            
            # # # Label smoothing experiment
            # label = (label * 0.9 + 0.05)
            # label[:,0] = 1
            # output = output.view(-1)
            #output = output.view(-1,output.shape[-1]) if self.is_2d else input.view(-1)
            loss = criterion(output, label)
            #loss = focal_loss(output, label)
            loss.backward()
            optimizer.step()

            init_loss += loss.item()

            #validating the model based on trained for a certain epcho

            



            if i_batch % 5 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.3f}'.format(
                    epoch, i_batch * len(x_train), len(train_generator.dataset),
                    100. * i_batch / len(train_generator), loss.item(), train_acc))

                ### Save the best validation loss model (use it for )   

        if True:
                accuracy = 0
                test_loss = 0
                model.eval() # Evaluation mode
                with torch.no_grad():
                    for images, labels in val_generator:
                        
                        
                        images = images.to(device, dtype=torch.float)
                        labels = labels.to(device)

                        output = model(images)
                        #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        #accuracy += pred.eq(labels.view_as(pred)).sum().item()
                        #accuracy += (torch.max(output, 1)[1].view(labels.size()) == labels).sum().item()
                        #accuracy += len((torch.max(output, 1)[1] == torch.max(labels.data, 1)[1]).nonzero())
                        accuracy += ordinal_accuracy(output, labels, 0.5)
                        test_loss += criterion(output, labels).item()
                        #print(test_loss)
                        #print(accuracy)
                        #e = time.time()

                        
                
                
                val_loss,accuracy = test_loss/len(val_generator.dataset), accuracy/len(val_generator.dataset)

                

                #val_loss,accuracy = validate(model, criterion, val_generator) 
                print("Epoch: {} of {}, ".format(epoch+1, num_epochs),
                        
                        "Test Loss: {:.3f}, ".format(val_loss),
                        "Val_Accuracy: %{:.1f}".format(accuracy*100))
                
                if accuracy > best_val_acc:
                    best_val_acc = accuracy
                    torch.save(model.state_dict(), "D:/MSCS/ADL/DL_Projects/saved_models/model_floss_apts_densenet_ot_"+str(epoch)+".pt")

                model.train() 
                #init_loss = 0




