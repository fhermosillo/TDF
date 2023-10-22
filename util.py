import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy
import numpy as np
import sklearn as sk
import os
from sklearn.metrics import matthews_corrcoef
import pickle


# def plot_confmat2(cm):
#     nrows=cm.shape[0]
#     ncols=cm.shape[1]
#     df_cm = pd.DataFrame(cm, range(nrows), range(ncols))
#     sn.set(font_scale=1.4) # for label size
#     sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
#     plt.show()
#     sn.reset_orig()

def plot_confmat(cm,normalized=False,classes=None,
                          title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #sn.reset_orig()
    #plot_confmat2(cm)
    #return
    # Only use the labels that appear in the data    
    if classes is None:
        classes = [f"Class {i}" for i in range(cm.shape[0])]
    cm2 = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 8))
    plt.imshow(cm2, interpolation='nearest', cmap=cmap)
    plt.xticks(np.arange(cm.shape[1]), classes,fontsize=20)
    plt.yticks(np.arange(cm.shape[0]), classes, rotation=90,fontsize=20)
    plt.ylabel('ACTUAL',fontsize=20)
    plt.xlabel('PREDICTED',fontsize=20)
    if title is not None:
        plt.title("$" + title + "$", fontsize=20)
    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             #rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if normalized:
        fmt = '.2f'
    else:
        fmt = 'd'
    thresh = cm2.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalized:
                plt.text(j, i, "$" + format(cm2[i,j], fmt) + "\%$",ha="center", va="center",color="white" if cm2[i, j] > thresh else "black", fontsize=20)
            else:
                plt.text(j, i, "$" + format(int(cm[i,j]), fmt) + "$",ha="center", va="center",color="white" if cm2[i, j] > thresh else "black", fontsize=20)
            
    plt.tight_layout()
    

def metrics_imbalance(cm, name = "", verbose=False):
    TP=cm[1,1]
    TN=cm[0,0]
    FN=cm[1,0]
    FP=cm[0,1]
    #FN=cm[0,1] #@Wrong see sklearn.metrics.confusion_matrix
    #FP=cm[1,0] #@Wrong see sklearn.metrics.confusion_matrix
    
    mn = TN + FP
    mp = FN + TP
    m = TP + TN + FP + FN
    delta = 2*(mp/m) - 1
    lambda_pp = TP/mp
    lambda_nn = TN/mn
    
    # Class imbalance metrics 
    SNS = lambda_pp
    SPC = lambda_nn
    PRC = lambda_pp*(1 + delta) / (lambda_pp*(1 + delta) + (1 - lambda_nn)*(1 - delta))
    NPV = lambda_nn*(1 - delta) / (lambda_nn*(1 - delta) + (1 - lambda_pp) * (1 + delta))
    ACC = (lambda_pp*(1 + delta)/2) + (lambda_nn*(1 - delta)/2)
    F1 = 2*lambda_pp*(1 + delta) / ((1 + lambda_pp) * (1 + delta) + (1 - lambda_nn) * (1 - delta))
    GM = np.sqrt(lambda_pp * lambda_nn)
    MCCn = 0.5*(( lambda_pp + lambda_nn - 1)/np.sqrt((lambda_pp + (1 - lambda_nn)*(1 - delta)/(1 + delta)) * (lambda_nn + (1 - lambda_pp)*(1 + delta)/(1 - delta))) + 1)
    BM = (lambda_pp + lambda_nn) / 2
    MK = 0.5*((1 + delta)/((1 + delta) + (1 - lambda_nn)*(1 - delta)/lambda_pp) + (1 - delta)/((1 - delta) + (1 - lambda_pp)*(1 + delta)/lambda_nn))
    
    # Class balance metrics
    SNSb = lambda_pp
    SPCb = lambda_nn
    PRCb = lambda_pp / (lambda_pp + (1 - lambda_nn))
    NPVb = lambda_nn / (lambda_nn + (1 - lambda_pp))
    ACCb = (lambda_pp + lambda_nn)/2
    F1b = 2*lambda_pp/(2 + lambda_pp - lambda_nn)
    GMb = np.sqrt(lambda_pp * lambda_nn)
    MCCb = 0.5*((lambda_pp + lambda_nn - 1)/np.sqrt((lambda_pp + (1 - lambda_nn)) * (lambda_nn + (1 - lambda_pp))) + 1)
    BMb = (lambda_pp + lambda_nn) / 2
    MKb = 0.5 * ((1 / (1 + (1 - lambda_nn)/lambda_pp)) + (1 / (1 + (1 - lambda_pp)/lambda_nn)))
    
    # Bias
    BSNS = 0
    BSPC = 0
    BPRC = ((1 + delta)/((1 + delta) + (1 - delta)*(1-lambda_nn)/lambda_pp)) - 1/(1 + (1 - lambda_nn)/lambda_pp)
    BNPV = ((1 - delta)/((1 - delta) + (1 + delta)*(1-lambda_pp)/lambda_nn)) - 1/(1 + (1 - lambda_pp)/lambda_nn)
    BACC = (delta/2)*(lambda_pp - lambda_nn)
    BF1 = ((2*lambda_pp*(1 + delta))/((1 + lambda_pp)*(1 + delta) + (1 - lambda_nn)*(1 - delta))) - ((2*lambda_pp)/(2 + lambda_pp - lambda_nn))
    BGM = 0
    BMCC = ((lambda_pp + lambda_nn - 1)/(2*np.sqrt((lambda_pp + (1 - lambda_nn)*(1-delta)/(1+delta)) * (lambda_nn + (1- lambda_pp)*(1 + delta)/(1 - delta))))) - ((lambda_pp + lambda_nn - 1)/(2*np.sqrt((lambda_pp + (1 - lambda_nn)) * (lambda_nn + (1 - lambda_pp)))))
    BBM = 0
    BMK = 0.5*(((1 + delta)/((1 + delta) + ((1 - lambda_nn)/(lambda_pp))*(1 - delta))) - ((1)/(1 + ((1 - lambda_nn)/(lambda_pp)))) + ((1 - delta)/((1 - delta) + ((1 - lambda_pp)/(lambda_nn))*(1 + delta))) - ((1)/(1 + ((1 - lambda_pp)/(lambda_nn)))))
    
    BalACC=0.5*(TP/(TP + FN) + TN/(TN + FP))
    
    col_names = ['$\mu(\lambda_{pp}, \lambda_{nn}, \delta)$', '$\mu_b(\lambda_{pp}, \lambda_{nn})$', '$B_\mu(\lambda_{pp}, \lambda_{nn}, \delta)$']
    row_names    = ['SNS', 'SPC', 'PRC', 'NPV', 'ACC', 'F1', 'GM', 'MCCn', 'BMn', 'MKn', 'BalACC']
    
    luque_matrix = np.reshape((SNS, SNSb, BSNS,
                               SPC, SPCb, BSPC, 
                               PRC, PRCb, BPRC,
                               NPV, NPVb, BNPV,
                               ACC, ACCb, BACC,
                               F1,  F1b,  BF1,
                               GM,  GMb,  BGM,
                               MCCn, MCCb, BMCC,
                               BM,  BMb,  BBM,
                               MK,  MKb,  BMK,
                               BalACC, BalACC, 0), (11, 3))
    luque_df = pd.DataFrame(luque_matrix, columns=col_names, index=row_names)
    return luque_df,delta,row_names
    
    









class SDCVDataset(Dataset):
    def __init__(self, X, Y):
        """SDCV Vehicle Dataset """
        """
        Args:
            X (DataFrame): Data
            Y (DataFrame): Labels
        """
        torch._assert(isinstance(X,list), "X must be a list of datasets")
        torch._assert(len(X)>=1, "X must contain at least one feature model")
        
        self.M = len(X)
        self.X = []
        for i in range(self.M):
            #torch.from_numpy(X[i]).float()
            self.X.append(torch.tensor(X[i],dtype=torch.float32,requires_grad=False))
        self.Y = torch.tensor(Y.values,dtype=torch.float32,requires_grad=False)
        
        self.labels = torch.unique(self.Y)
        
    def __len__(self):
        return self.X[0].shape[0]
        
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_out = []
        for i in range(self.M):
            data_out.append(self.X[i][idx,:])
        data_out.append(self.Y[idx])
        
        return data_out
    
    def getAll(self):
        return self.X + [self.Y]
    
    def getNumModals(self):
        return self.M
    
    def getModals(self):
        return self.X
    
    def getTarget(self):
        return self.Y
    
    def getLabels(self):
        return self.labels





class MTLDataset(Dataset):
    def __init__(self, X, Y):
        """SDCV Vehicle Dataset """
        """
        Args:
            X (DataFrame): Data
            Y (DataFrame): Labels
        """
        torch._assert(isinstance(X,list), "X must be a list of datasets")
        torch._assert(len(X)>=1, "X must contain at least one feature model")
        
        self.M = len(X)
        self.X = []
        
        for i in range(self.M):
            self.X.append(torch.tensor(X[i],dtype=torch.float32,requires_grad=False))
            #self.X.append(torch.tensor(X[i].values,dtype=torch.float32,requires_grad=False))
        tmp = torch.tensor(Y.values,dtype=torch.int,requires_grad=False)
        self.Y=[]
        self.mlabel=True
        self.labels=[]
        for i in range(tmp.shape[1]):
            nclass = torch.unique(tmp[:,i])
            if (nclass< 0).any():
                nclass+=1
                self.Y.append(torch.eye(len(nclass))[tmp[:,i] +1])
            else:
                self.Y.append(torch.eye(len(nclass))[tmp[:,i]])
            
            tmp2=[]
            for n in range(len(nclass)):
                tmp2.append(n)
            self.labels.append(tmp2)
        if tmp.shape[1]==1:
            self.Y=self.Y[0]
            self.mlabel=False
        
    def __len__(self):
        return self.X[0].shape[0]
        
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_out = []
        for i in range(self.M):
            data_out.append(self.X[i][idx,:])
        if self.mlabel:
            for i in range(len(self.Y)):
                data_out.append(self.Y[i][idx,:])
        else:
            data_out.append(self.Y[idx,:])
        
        return data_out
    
    def getAll(self):
        return self.X + self.Y
    
    def getNumModals(self):
        return self.M
    
    def getModals(self):
        return self.X
    
    def getTarget(self):
        return self.Y
    
    def getLabels(self):
        return self.labels






def load_sdcv(filename, xlabels, ylabel, rate=0.15, verbose=False):
    # parse the dataset
    ds = pd.read_csv(filename)
    
    # Plot sample count
    # fpath = os.path.split(filename)
    # videoname = fpath[1][0:-4]
    # sns.countplot(x = ylabel[0], data=ds)
    # plt.title('Video ' + videoname)
    # plt.show()
    
    # Split datasets
    Y=ds.loc[:,ylabel]
    if verbose:
        PClass=len(ds[ds.TARGET==1])
        NClass=len(ds[ds.TARGET==0])
        print(PClass+NClass, "(", PClass, "/" , NClass, ")")
        #print('NNegClass: ', NClass)
        #print('NPosClass: ', PClass)
    else:
        ds_train, ds_test = train_test_split(ds, train_size=rate, shuffle=True, stratify=Y)
        Y=ds_test.loc[:,ylabel]
        ds_test, ds_valid = train_test_split(ds_test, train_size=0.6, shuffle=True, stratify=Y)
        
        # Training dataset
        M = len(xlabels)
        scalers = [None]*M
        X = []
        for i in range(M):
            X.append(ds_train.loc[:,xlabels[i]])
            scalers[i] = StandardScaler()
            scalers[i].fit_transform(X[-1])
            # print("Scaler #", i)
            # print("\tMean: ", scalers[i].mean_)
            # print("\tVar: ", scalers[i].var_)
            X[-1] = scalers[i].transform(X[-1])
        Y = ds_train.loc[:,ylabel]
        train_set = SDCVDataset(X,Y)
        
        # Validation dataset
        X = []
        for i in range(M):
            X.append(ds_valid.loc[:,xlabels[i]])
            X[-1] = scalers[i].transform(X[-1])
        Y = ds_valid.loc[:,ylabel]
        valid_set = SDCVDataset(X,Y)
        
        # Testing dataset
        X = []
        for i in range(M):
            X.append(ds_test.loc[:,xlabels[i]])
            X[-1] = scalers[i].transform(X[-1])
        Y = ds_test.loc[:,ylabel]
        test_set = SDCVDataset(X,Y)
        
        # Dimensions
        input_dims = []
        for i in range(M):
            input_dims.append(X[i].shape[1])
    
        return train_set, valid_set, test_set, input_dims, scalers


def load_sdcv_cv(filename, xlabels, ylabel):
    # parse the dataset
    ds = pd.read_csv(filename)
    
    # fpath = os.path.split(filename)
    # videoname = fpath[1][0:-4]
    
    # Split datasets
    Y=ds.loc[:,ylabel]
    ds_train, ds_test = train_test_split(ds, train_size=0.99, shuffle=True, stratify=Y)
    
    # Training dataset
    M = len(xlabels)
    X = []
    for i in range(M):
        X.append(torch.tensor(np.concatenate((ds_train.loc[:,xlabels[i]],ds_test.loc[:,xlabels[i]]))))
    Y = pd.DataFrame(data=np.concatenate((ds_train.loc[:,ylabel],ds_test.loc[:,ylabel])) )
    ds = SDCVDataset(X,Y)
    
    # Dimensions
    input_dims = []
    for i in range(M):
        input_dims.append(X[i].shape[1])

    return ds, input_dims


def load_sdcv_mtl(filename, xlabels, ylabels, train_sz=0.5):
    # parse the dataset
    ds = pd.read_csv(filename)
    
    
    # Split datasets
    Y=ds.loc[:,ylabels]
    
    
    
    
    ax,N=sns.countplot(x = ylabels[0], data=ds)
    plt.title("Classification")
    plt.show()
    sns.countplot(x = ylabels[1], data=ds)
    plt.title("Occlusion")
    plt.show()
    
    
    
    ds_train, ds_test = train_test_split(ds, train_size=train_sz, shuffle=True, stratify=Y)
    Y=ds_test.loc[:,ylabels]
    ds_test, ds_valid = train_test_split(ds_test, train_size=0.5, shuffle=True, stratify=Y)
    
    # Training dataset
    M = len(xlabels)
    scalers = [None]*M
    X = []
    for i in range(M):
        X.append(ds_train.loc[:,xlabels[i]])
        scalers[i] = StandardScaler()
        scalers[i].fit_transform(X[-1])
        X[-1] = scalers[i].transform(X[-1])
    Y = ds_train.loc[:,ylabels]
    train_set = MTLDataset(X,Y)
    
    # Validation dataset
    X = []
    for i in range(M):
        X.append(ds_valid.loc[:,xlabels[i]])
        X[-1] = scalers[i].transform(X[-1])
    Y = ds_valid.loc[:,ylabels]
    valid_set = MTLDataset(X,Y)
    
    # Testing dataset
    X = []
    for i in range(M):
        X.append(ds_test.loc[:,xlabels[i]])
        X[-1] = scalers[i].transform(X[-1])
    Y = ds_test.loc[:,ylabels]
    test_set = MTLDataset(X,Y)
    
    # Dimensions
    input_dims = []
    for i in range(M):
        input_dims.append(X[i].shape[1])

    return train_set, valid_set, test_set, input_dims, scalers





def load_sdcv_crossval(filename, xlabels, ylabel):
    # parse the dataset
    ds = pd.read_csv(filename)
    
    # Split datasets
    Y=ds.loc[:,ylabel]
    
    # Training dataset
    M = len(xlabels)
    X = []
    for i in range(M):
        X.append(ds.loc[:,xlabels[i]].values)
    dataset = SDCVDataset(X,Y)
    
    # Dimensions
    input_dims = []
    for i in range(M):
        input_dims.append(X[i].shape[1])

    return dataset,input_dims





class Serialize():
    def __init__(self):
        1
        
    def save(self,obj,filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    
    def load(self,filename):
        with open(filename, 'rb') as inp:
            return pickle.load(inp)


class SynthFusionDataset(Dataset):
    def __init__(self, X1, X2, Y):
        """SDCV Synthetic Vehicle Dataset """
        """
        Args:
            X (DataFrame): Data
            Y (DataFrame): Labels
        """
        self.X1 = torch.Tensor(X1)
        self.X2 = torch.Tensor(X2)
        self.Y = torch.Tensor(Y)
    
    def __len__(self):
        return self.X1.shape[0]
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return [self.X1[idx,:], self.X2[idx,:], self.Y[idx]]
    
    def getAll(self):
        return [self.X1,self.X2,self.Y]
    
def load_synthetic():
    # Dataset creation
    X1n=torch.randn((4000,5))
    X1p=torch.randn((4000,5))*0.1 + torch.Tensor([1000,8000,12000,7000,8000])
    X1=torch.cat((X1n,X1p), 0)
    
    X2n=torch.randn((4000,2))*0.1 + torch.Tensor([-2,1])
    X2p=torch.randn((4000,2))*0.03 + torch.Tensor([1500,1600])
    X2=torch.cat((X2n,X2p), 0)
    
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    X1 = scaler1.fit_transform(X1)
    X2 = scaler2.fit_transform(X2)
    
    X = torch.cat((torch.Tensor(X1),torch.Tensor(X2)), 1)
    
    Yn = torch.zeros((4000,1))
    Yp = torch.ones((4000,1))
    Y=torch.cat((Yn,Yp), 0)
    
    ds = torch.cat((X,Y), 1)
    
    # Train test split
    ds_train, ds_test = train_test_split(ds, train_size=0.8)
    ds_train, ds_valid = train_test_split(ds_train, train_size=0.8)
    
    
    # Dataset class
    X1 = ds_train[:,0:5]
    X2 = ds_train[:,5:7]
    Y = ds_train[:,-1]
    train_set = SynthFusionDataset(X1,X2,Y)
    
    X1 = ds_valid[:,0:5]
    X2 = ds_valid[:,5:7]
    Y = ds_valid[:,-1]
    valid_set = SynthFusionDataset(X1,X2,Y)
    
    X1 = ds_test[:,0:5]
    X2 = ds_test[:,5:7]
    Y = ds_test[:,-1]
    test_set  = SynthFusionDataset(X1,X2,Y)
    
    input_dims = (5, 2)
   
    return train_set, valid_set, test_set, input_dims