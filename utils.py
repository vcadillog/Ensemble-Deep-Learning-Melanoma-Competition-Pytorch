import torch as th
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import glob
from PIL import Image


class EnsClassificationDataset(Dataset):
    def __init__(self, X,y,class_type):
        self.dir ='/kaggle/input/jpeg-melanoma-256x256/'+class_type        
        self.X = X    
        self.y = th.from_numpy(y.to_numpy())   
        self.class_type = class_type     

    def __len__(self):
        return len(self.X)    

    def __getitem__(self, idx):

              
        img = glob.glob(self.dir+'/'+self.X[idx][0]+'.jpg')        
        X_tab = th.from_numpy(self.X[idx][1:].astype(float))

        X_img = Image.open(img[0])
        X_img = ToTensor()(X_img)                     
        if self.class_type == 'train':
            y = self.y[idx]    
            return X_img,X_tab.to(th.float32),y.type(th.LongTensor)
        else:
            return X_img,X_tab.to(th.float32)

def dataloader(data_train,data_val,data_test,batch_size,shuffle = True, drop_last = False):
    train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    test_dataloader = DataLoader(data_test, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    val_dataloader = DataLoader(data_val, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)   

    train_img , train_tabular, train_y = next(iter(train_dataloader))
    print(f"Image batch shape: {train_img.size()}")
    print(f"Tabular batch shape: {train_tabular.size()}")
    print(f"Classes batch shape: {train_y.size()}")
    return train_dataloader,val_dataloader,test_dataloader,train_tabular.size()

class EnsembleNetwork(nn.Module):
    def __init__(self,input_dim: int) -> None:
        super(EnsembleNetwork, self).__init__()    
        self.fc1 = nn.Linear(input_dim, 18)        
        self.fc2 = nn.Linear(18, 36)
        self.fc3 = nn.Linear(36, 18)
        self.fc4 = nn.Linear(18,input_dim)
        self.output = nn.Linear(input_dim+1000, 1) # Size Output of resnet: 1000
        self.relu = nn.ReLU()           
         
        self.resnet = th.hub.load('pytorch/vision:v0.10.0', 'resnet18')

    def forward(self, img: th.Tensor, tab: th.Tensor) -> th.Tensor:              
        
        tab = self.fc1(tab)
        tab = self.relu(tab)
        tab = self.fc2(tab)
        tab = self.relu(tab)
        tab = self.fc3(tab)
        tab = self.relu(tab)
        tab = self.fc4(tab)
        tab = self.relu(tab)
        
        img = self.resnet(img) 
        
        x = th.cat((img,tab), 1)        
        x = self.output(x)
        x = self.relu(x)
        
        return x    
    

class learning_loop():
    def __init__(self, N_EPOCH,model,device,train_data,val_data,test_data,optimizer,criterion, directory,mode ) -> None:
        self.N_EPOCH = N_EPOCH
        self.model = model
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.criterion = criterion        
        self.mode = mode     
        self.writer1=SummaryWriter('logs/'+directory+'/train',flush_secs=60)
        self.writer2=SummaryWriter('logs/'+directory+'/val',flush_secs=60)   

    def train(self):
        for epoch in range(1, self.N_EPOCH+1):
            for (_,train) , (_,val) in zip(enumerate(self.train_data, 0),enumerate(self.val_data, 0)):
                self.model.train()
                X_img_train_ts,X_tab_train_ts, y_train_ts = train
                X_img_val_ts, X_tab_val_ts,y_val_ts = val

                X_img_train_ts,X_tab_train_ts, y_train_ts  = X_img_train_ts.to(self.device),X_tab_train_ts.to(self.device), y_train_ts.to(self.device)
                X_img_val_ts, X_tab_val_ts,y_val_ts = X_img_val_ts.to(self.device), X_tab_val_ts.to(self.device),y_val_ts.to(self.device) 
                

                out = self.model(X_img_train_ts,X_tab_train_ts)      
                loss = self.criterion(out, y_train_ts)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.mode == 'classification':
                    acc = (th.argmax(out, dim=1) == y_train_ts).float().mean().item()
                
                self.model.eval()
                with th.no_grad():
                    out_val = self.model(X_img_val_ts,X_tab_val_ts)
                    loss_val = self.criterion(out_val, y_val_ts)
                    if self.mode == 'classification':
                        acc_val = (th.argmax(out_val, dim=1) == y_val_ts).float().mean().item()
               
            self.writer1.add_scalar('Loss', loss.item(), epoch)
            self.writer2.add_scalar('Loss', loss_val.item(), epoch)  
                       
            if self.mode == 'classification':
                self.writer1.add_scalar('Accuracy', acc*100 , epoch)
                self.writer2.add_scalar('Accuracy', acc_val*100, epoch)                           

            if epoch % 20 == 0:
                    if self.mode == 'classification':
                        print('Epoch : {:3d} / {}, Loss : {:.4f}, Accuracy : {:.2f} %, Val Loss : {:.4f}, Val Accuracy : {:.2f} %'.format(
                            epoch, self.N_EPOCH, loss.item(), acc*100, loss_val.item(), acc_val*100))
                    else:
                        print('Epoch : {:3d} / {}, Loss : {:.4f},  Val Loss : {:.4f}'.format(
                            epoch, self.N_EPOCH, loss.item(), loss_val.item()))
                
        self.writer1.close()        
        self.writer2.close()        

    # def test(self):
    #     acc_avg = 0
    #     loss_avg = 0
    #     y_list = [] # i,y_test,y_pred
        
    #     for (i,test) in enumerate(self.test_data, 0):
            
    #         X_test_ts, y_test_ts = test           
    #         X_test_ts, y_test_ts = X_test_ts.to(self.device), y_test_ts.to(self.device)

            
    #         self.model.eval()
    #         with th.no_grad():
    #             out_test = self.model(X_test_ts.to(th.float32))
    #             loss_test = self.criterion(out_test, y_test_ts)
    #             if self.mode == 'classification':
    #                 acc_test = (th.argmax(out_test, dim=1) == y_test_ts).float().mean().item()
            
    #         if self.mode == 'classification':
    #             acc_avg += acc_test            
    #         else:
    #             y_list.append([i,y_test_ts.cpu().numpy(),out_test.cpu().numpy()])                
                
    #         loss_avg += loss_test
    #     if self.mode == 'classification':
    #         print('Val avg Loss : {:.4f}, Val avg Accuracy : {:.2f} %'.format(
    #             loss_avg/i,acc_avg/i*100))  
    #     else:
    #         print('Val avg Loss : {:.4f}'.format(
    #             loss_avg/i))  
    #         return     


    