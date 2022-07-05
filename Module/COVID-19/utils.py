import torch
import numpy as np
import seaborn as sns
import prettytable
import matplotlib.pyplot as plt
from thop.profile import profile
import time

class  ImageShow(object):
    def __init__(self,
                train_loss_list,train_acc_list,
                test_loss_list,test_acc_list,
                test_auc_list,
                val_loss_list,val_acc_list):        
        self.trainll, self.trainacl = train_loss_list, train_acc_list
        self.testll, self.testacl = test_loss_list, test_acc_list
        self.testauc = test_auc_list
        self.valll, self.valacl = val_loss_list, val_acc_list
        
    def train(self,opt="Loss",write=True,custom_path=None,img_title=None,suf=None):
        if opt == 'Acc':
            img_portray(opt=opt,write=write,dates=self.trainacl,
                        #location='upper left',
                        label='Train_Acc',col='red',
                        img_title=img_title,suf=suf)
        elif opt == 'Loss':
            img_portray(opt=opt,write=write,dates=self.trainll,
                        #location='upper right',
                        linestyle="--",
                        label='Train_Loss',col='green',
                        img_title=img_title,suf=suf)
        if write:
            save_images(img_title=img_title,suf=suf,opt=opt)
        plt.show()
        
    def test(self,opt='Acc',write=True,custom_path=None,img_title=None,suf=None,**kwargs):
        if opt == 'Acc':
            img_portray(opt=opt,write=write,dates=self.testacl,
                        #location='upper left',
                        label='Test_Acc',col='red',
                        img_title=img_title,suf=suf)
        elif opt == 'Loss':
            img_portray(opt=opt,write=write,dates=self.testll,
                        #location='upper right',
                        linestyle="-.",
                        label='Test_Loss',col='green',
                        img_title=img_title,suf=suf)
        if write:
            save_images(split='test',img_title=img_title,suf=suf,opt=opt)
        plt.show()
        
    def val(self,opt='Acc',write=True,custom_path=None,img_title=None,suf=None):
        if opt == 'Acc':
            img_portray(opt=opt,write=write,dates=self.valacl,
                        linestyle="dotted",col='red',
                        label='Val_Acc',#location='upper left',
                        img_title=img_title,suf=suf)
        elif opt == 'Loss':
            img_portray(opt=opt,write=write,dates=self.valll,
                        linestyle="-.",col='green',
                        label='Val_Loss',#location='upper right',
                        img_title=img_title,suf=suf)
        if write:
            save_images(split='Val',img_title=img_title,suf=suf,opt=opt)
        plt.show()
        
    def conclusion(self,opt="test",img_title=None):
        if opt == "test" and len(self.testacl) != 0:
            print(f'\033[31m=================Conclusion====================\033[0m')
            best_idx = self.testacl.index(max(self.testacl))
            # val_idx = (best_idx+1)-1
            best_epoch = (best_idx+1)
            print(f"Dataset:[\033[1;31m{img_title}\033[0m]")
            print(f"Best_Epoch [\033[1;31m{best_epoch}\033[0m]")
            # print("[Train] loss {self.trainll[best_epoch-1]};")
            print(f"[Test] \033[31mACC:{round(float(self.testacl[best_idx]),2)}%\033[0m.")
            # Loss:{self.testll[best_idx]}, AUC:{round(float(self.testauc[best_idx]),2)}%
            # print(f"[Test]:\033[32mVal_ACC:{round(float(max(self.testauc)),2)}%\033[0m.")
        if opt == "val" and len(self.valacl) != 0:
            print(f'\033[31m=================Conclusion====================\033[0m')
            best_idx = self.valacl.index(max(self.valacl))
            best_epoch = (best_idx+1)
            print(f"Dataset:[\033[1;31m{img_title}\033[0m]")
            print(f"Best_Epoch [\033[1;31m{best_epoch}\033[0m]")
            print(f"[Val] \033[31mACC:{round(float(self.valacl[best_idx]),2)}%\033[0m.")

        if opt == "auc" and len(self.testauc) != 0:
            print(f'\033[31m=================Conclusion====================\033[0m')
            best_idx = self.testauc.index(max(self.testauc))
            val_idx = (best_idx+1)-1
            best_epoch = (best_idx+1)
            print(f"Dataset:[\033[1;31m{img_title}\033[0m]")
            print(f"Best_Epoch [\033[1;31m{best_epoch}\033[0m]\n[Train] loss:{self.trainll[best_epoch-1]};")
            print(f"[Test] Loss:{self.testll[best_idx]}, \033[32mACC:{round(float(self.testacl[best_idx]),2)}%\033[0m.")
            print(f"[Test]:\033[32m AUC:{round(float(self.testauc[best_idx]),2)}%\033[0m.")
    
def img_portray(opt='Acc',write=True,
                split=None,custom_path='./tmp',
                dates=None,linestyle="dotted",
                label=None,col=None,location='best',
                img_title=None,suf=None):
    plt.style.use("seaborn")
    plt.title(img_title)
    plt.xlabel("Epochs")
    if opt == 'Acc' and dates != None:
        epoch_nums = np.arange(len(dates))
        y = dates
        img_max = np.argmax(y)
        show_max = round(float(y[img_max]),2)
        plt.plot(img_max,show_max ,'8')
        plt.annotate(show_max,xy=(img_max,show_max),xytext=(img_max,show_max))
        plt.plot(epoch_nums, y, linestyle=linestyle,c=col,label=label)
        plt.ylabel("Accuracy")
        plt.legend(loc=location)
    elif opt == 'Loss' and dates != None:
        epoch_nums= np.arange(len(dates))
        y = torch.tensor(dates,device='cpu')      
        img_min = np.argmin(y)
        show_min = round(float(y[img_min]),6)
        
        plt.annotate(show_min,xy=(img_min,show_min),xytext=(img_min,show_min))
        plt.plot(img_min,show_min,'8')     
        plt.plot(epoch_nums, y,linestyle=linestyle,c=col,label=label)
        plt.ylabel("Loss")
        plt.legend(loc=location)
    else:
        print("Please input the right decision.")
    
def save_images(split='train',custom_path='./tmp',img_title=None,suf=None,opt=None):
    if split == 'train':
        plt.savefig(f'{custom_path}/{img_title}/{suf}/{split}_{opt}.png',dpi=300)
    else:
        plt.savefig(f'{custom_path}/{img_title}/{suf}/{split}_{opt}.png',dpi=300)

def draw_size_acc(data_dict,custom_path='./tmp',img_title=None,suf=None,opt=None):
    sx=[]
    sy=[]

    for i in range(len(data_dict)):
        x=sorted(data_dict.items(), key=lambda x: x[0])[i][0]
        y=sorted(data_dict.items(), key=lambda x: x[0])[i][1]
        sx.append(x)
        sy.append(y)

    plt.style.use("seaborn")
    plt.plot(sx, sy,label="Test_Data")
    plt.ylabel("Accuracy")
    plt.xlabel("Image_Size")
    plt.legend(loc="best") 
       
    plt.savefig(f'{custom_path}/{img_title}/{suf}/Size_Accuracy.png',dpi=300)

def confusion_matrix(evl_result,n_cla,cla_dict,data,img_title=None,suf=None):
    plt.figure(figsize=(12,9))
    sb = range(n_cla)
    sns.heatmap(evl_result,annot=True,cmap="Blues",cbar=True,fmt="g", annot_kws={"size": 20})
    plt.yticks([index + 0.5 for index in sb],cla_dict.values(),fontsize=16)
    plt.xticks([index + 0.5 for index in sb],cla_dict.values(),fontsize=16)

    plt.title("Confusion Matrix",fontsize=24)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=16)

    # vname = lambda v,nms: [ vn for vn in nms if id(v)==id(nms[vn])][0]
    # kn = vname(evl_result,locals())
    if evl_result.sum().item() == len(data):
        kn = 'test'
    else:
        kn = 'val'

    plt.savefig(f"./tmp/{img_title}/{suf}/Confusion_Matrix_{kn}.png",dpi=300)
    
def metrics_scores(evl_result,n_classes,cla_dict):
    result_table = prettytable.PrettyTable()
    result_table.field_names = ['Type','Precision', 'Recall', 'F1','Accuracy']    
    accuracy = float(torch.sum(evl_result.diagonal())/torch.sum(evl_result))  
    for i in range(n_classes):
        pre = float(evl_result[i][i] / torch.sum(evl_result,0)[i])
        recall = float(evl_result[i][i] / torch.sum(evl_result,1)[i])
        F1 = pre * recall * 2 / (pre + recall + 1e-8)
        result_table.add_row([cla_dict[i], round(pre, 4), round(recall, 3), round(F1, 3)," "])

    result_table.add_row(["Total:", " ", " ", " ",round(accuracy,4)])
    print(result_table)

def one_hot(x, length):
    batch_size = x.size(0)
    x_one_hot = torch.zeros(batch_size, length)
    for i in range(batch_size):
        x_one_hot[i, x[i]] = 1.0
    return x_one_hot

def pff(m_name,model,inputes):
    result_table = prettytable.PrettyTable()
    result_table.field_names = ['Model','Params(M)', 'FLOPs(G)', 'FPS']  

    total_ops, total_params = profile(model, (inputes,), verbose=False)
    Params = total_params / (1000 ** 2)
    ops = total_ops / (1000 ** 3)
    
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        output= model(inputes)
        torch.cuda.synchronize()
        end = time.time()
        single_fps = 1/(end-start)
        
    result_table.add_row([m_name, round(Params, 2), round(ops, 2), round(single_fps, 2)])
    print(result_table)