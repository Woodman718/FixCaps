import torch
import numpy as np
import matplotlib.pyplot as plt

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
        
    def conclusion(self,opt="Acc",img_title=None):
        if opt == "Acc" and len(self.testacl) != 0:
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
        if opt == "Auc" and len(self.testauc) != 0:
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
        plt.savefig(f'{custom_path}/{img_title}/{suf}/{split}_{opt}.png',dpi=100)
    else:
        plt.savefig(f'{custom_path}/{img_title}/{suf}/{split}_{opt}.png',dpi=100)
        
# def test(self,opt='acc',write=True,custom_path=None,img_title=None,suf=None):
#     plt.style.use("seaborn")
#     plt.title(img_title)
#     plt.xlabel("Epochs")
#     if opt == 'acc':
#         epoch_num = np.arange(len(self.testacl))
#         tacc = self.testacl
#         # tauc = self.testauc
#         #show max_value min_value
#         # img_min = np.argmin(tacc)
#         img_max = np.argmax(tacc)
#         # show_min = round(float(tacc[img_min]),2)
#         show_max = round(float(tacc[img_max]),2)
#         # plt.plot(img_min,tacc[img_min],'8')
#         plt.plot(img_max,show_max ,'8')
#         # plt.annotate(show_min,xy=(img_min,tacc[img_min]),xytext=(img_min,tacc[img_min]))
#         plt.annotate(show_max,xy=(img_max,show_max),xytext=(img_max,show_max))
#         #
#         # val_min = np.argmin(vacc)
#         # auc_max = np.argmax(tauc)
#         # show_val_min = round(float(vacc[val_min]),2)
#         # show_auc_max = round(float(tauc[auc_max]),2)
#         # plt.plot(val_min,vacc[val_min],'8')
#         # plt.plot(auc_max,show_auc_max,'8')
#         # plt.annotate(show_val_min,xy=(val_min,vacc[val_min]),xytext=(val_min,vacc[val_min]))
#         # plt.annotate(show_auc_max,xy=(auc_max,show_auc_max),xytext=(auc_max,show_auc_max))
#         #end
#         plt.plot(epoch_num, self.testacl,c="red",label='Test_Acc')
#         # plt.plot(epoch_num, tauc,c="blue",label='Test_Auc', linestyle='--')
#         plt.ylabel("Accuracy")
#         plt.legend(loc='upper left')
#     if opt == 'loss':
#         tloss = torch.tensor(self.testll,device='cpu')
#         # val_loss = torch.tensor(self.valll,device='cpu')
#         epoch_num = np.arange(len(self.testll))
#         #show max_value min_value
#         img_min = np.argmin(tloss)
#         show_min = round(float(tloss[img_min]),6)
#         plt.plot(img_min,show_min,'8')
#         plt.annotate(show_min,xy=(img_min,show_min),xytext=(img_min,show_min))
#         #
#         # val_min = np.argmin(val_loss)
#         # show_val_min = round(float(val_loss[val_min]),6)
#         # plt.plot(val_min,show_val_min,'8')
#         # plt.annotate(show_val_min,xy=(val_min,show_val_min),xytext=(val_min,show_val_min))
#         #end
#         plt.plot(epoch_num, tloss,label='Test_Loss')
#         # plt.plot(epoch_num, val_loss, label='Val_Loss',marker = "s",markersize=8, linestyle='--')
#         # plt.plot(epoch_num, val_loss, c="blue",label='Val_Loss', linestyle='--')
#         plt.ylabel("Average loss")
#         plt.legend(loc='upper right')

#     if write:
#         save_images(split='test',img_title=img_title,suf=suf,opt=opt)
#     # plt.ylim(0, 1.0)
#     plt.show()

# def val(self,opt='acc',write=True,custom_path=None,img_title=None,suf=None):
#     #markers = {'Test_Acc': 'o', 'Test_Loss': 's'}
#     plt.style.use("seaborn")
#     plt.title(img_title)
#     plt.xlabel("Epochs")

#     if opt == 'acc':
#         epoch_num = np.arange(len(self.testacl))
#         tacc = self.testacl
#         tauc = self.valacl
#         #show max_value min_value
#         # img_min = np.argmin(tacc)
#         img_max = np.argmax(tacc)
#         # show_min = round(float(tacc[img_min]),2)
#         show_max = round(float(tacc[img_max]),2)
#         # plt.plot(img_min,tacc[img_min],'8')
#         plt.plot(img_max,show_max ,'8')
#         # plt.annotate(show_min,xy=(img_min,tacc[img_min]),xytext=(img_min,tacc[img_min]))
#         plt.annotate(show_max,xy=(img_max,show_max),xytext=(img_max,show_max))
#         #
#         # val_min = np.argmin(vacc)
#         auc_max = np.argmax(tauc)
#         # show_val_min = round(float(vacc[val_min]),2)
#         show_auc_max = round(float(tauc[auc_max]),2)
#         # plt.plot(val_min,vacc[val_min],'8')
#         plt.plot(auc_max,show_auc_max,'8')
#         # plt.annotate(show_val_min,xy=(val_min,vacc[val_min]),xytext=(val_min,vacc[val_min]))
#         plt.annotate(show_auc_max,xy=(auc_max,show_auc_max),xytext=(auc_max,show_auc_max))
#         #end
#         plt.plot(epoch_num, self.testacl,c="red",label='Test_Acc')
#         plt.plot(epoch_num, tauc,c="blue",label='Val_Acc', linestyle='--')
#         plt.ylabel("Accuracy")
#         plt.legend(loc='lower right')
#     if opt == 'loss':
#         tloss = torch.tensor(self.testll,device='cpu')
#         val_loss = torch.tensor(self.valll,device='cpu')
#         epoch_num = np.arange(len(self.testll))
#         #show max_value min_value
#         img_min = np.argmin(tloss)
#         show_min = round(float(tloss[img_min]),6)
#         plt.plot(img_min,show_min,'8')
#         plt.annotate(show_min,xy=(img_min,show_min),xytext=(img_min,show_min))
#         #
#         val_min = np.argmin(val_loss)
#         show_val_min = round(float(val_loss[val_min]),6)
#         plt.plot(val_min,show_val_min,'8')
#         plt.annotate(show_val_min,xy=(val_min,show_val_min),xytext=(val_min,show_val_min))
#         #end
#         plt.plot(epoch_num, tloss,label='Test_Loss')
#         plt.plot(epoch_num, val_loss, label='Val_Loss',marker = "s",markersize=8, linestyle='--')
#         # plt.plot(epoch_num, val_loss, c="blue",label='Val_Loss', linestyle='--')
#         plt.ylabel("Average loss")
#         plt.legend(loc='upper right') 

#     if write:
#         save_images(split='val',img_title=img_title,suf=suf,opt=opt)
#     # plt.ylim(0, 1.0)
#     plt.show()