# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 02:59:30 2023

@author: User123
"""


import torch
import numpy as np
import os
import datetime
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import sklearn as sk
from NetworkModelV6 import NetworkModel
from util import load_sdcv_mtl
from util import metrics_imbalance
import time
import pickle
from collections import namedtuple



MyStruct = namedtuple("MyStruct", "nmodels vars ranks J")

def print_metrics(modals,target,mdl,nmodals=1, device="cpu"):
    if nmodals==2:
        Yhat = mdl(modals[0].to(device),modals[1].to(device))
    elif nmodals == 1:
        Yhat = mdl(modals[0].to(device))
        
    y_true = target.cpu().data.numpy()
    y_pred = np.round(Yhat.cpu().data.numpy())
    cm  = sk.metrics.confusion_matrix(y_true, y_pred)
    
    
    
    
    luque_df,delta = metrics_imbalance(cm,False,"")
    
    return luque_df





def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def read_object(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)
    


    

def main():
    # External Parameters
    Ranks=[-1,1,2,3,4]
    Nreps = 1
    FusedDim=[2,4,8,10]
    train_test_ratio = 0.5
    epochs = 200
    batch_sz = 64
    patience=30
    
    # Filenames ---------------------------------------------------------------
    filenames=[]
    filenames.append('D:\CINVESTAV\Matlab\Occlusion\OCC-CLASS-SAM_2.csv')
    
    
    ylabels=['YCLASS','YOCC']
    xlabels=[]
    xlabels.append(['WIDTH','SOLIDITY','ORIENTATION','ECCENTRICITY','COMPACTNESS_RATIO'])
    xlabels.append(['WIDTH','AREA','ASPECT_RATIO'])
    xlabels.append(['CENTROID_X','CENTROID_Y'])
    
    # Training Parameters -----------------------------------------------------
    params = dict()
    params['factor_learning_rate'] = [0.0003, 0.0005, 0.001, 0.003]
    params['learning_rate'] = [0.000003, 0.00005, 0.0001, 0.003]
    params['weight_decay'] = [0.001, 0.002, 0.005, 0.008]
    
    factor_lr = np.random.choice(params['factor_learning_rate'])
    lr = np.random.choice(params['learning_rate'])
    decay = np.random.choice(params['weight_decay'])
    
    
    # Outfiles ----------------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA support is enabled")
    
    
    # Set output directory
    dirpath  = str(datetime.date.today())
    n = 1
    tmp_dir = dirpath
    while True:
        if os.path.isdir(tmp_dir) == False:
            break
        else:
            tmp_dir = dirpath + "-" + str(n)
            n = n + 1
    dirpath = tmp_dir
    os.mkdir(dirpath)
    
    # Save model information
    with open(dirpath + '/settings.txt', 'w') as f:
        k = 1
        f.write("Iters=" + str(Nreps) + "\n")
        f.write("LR=" + str(lr) + "\n")
        f.write("FLR=" + str(factor_lr) + "\n")
        f.write("DEC=" + str(decay) + "\n")
        f.write("M=" + str(len(xlabels)) + "\n")
        f.write("J=" + str(FusedDim) + "\n")
        f.write("R=" + str(Ranks) + "\n")
        for xlabel in xlabels:
            f.write("FS" + str(k) + "=[\t")
            k = k + 1
            for varname in xlabel:
                f.write(f"{varname}\t")
            f.write("]\n")
        
    
    m = MyStruct(len(xlabels), xlabels, Ranks, FusedDim)
    p=[]
    p.append(m)
    save_object(p, dirpath + '/settings.bin')
    
    # -------------------------------------------------------------------------
    # START TRAINING ----------------------------------------------------------
    # -------------------------------------------------------------------------
    print("Output Directory: ", dirpath)
    try:
        for ivideo in range(len(filenames)):
            filename = filenames[ivideo]
            fpath = os.path.split(filename)
            mtdf_metrics = []   # Confusion matrix
            nmodals = len(xlabels)
            print("VIDEO: ", fpath[1][0:-4])
            start = time.time()
            for nrep in range(Nreps):
                # Dataset loading ---------------------------------------------------------
                train_set, valid_set, test_set,input_dims,scaler = load_sdcv_mtl(filename, xlabels, ylabels, train_test_ratio)
                I=input_dims
                nmodals = train_set.getNumModals()
                
                train_iterator = DataLoader(train_set, batch_size=int(batch_sz), shuffle=True)
                valid_iterator = DataLoader(valid_set, batch_size=int(batch_sz), shuffle=True)
                test_iterator  = DataLoader(test_set,  batch_size=int(batch_sz), shuffle=True)
                
                labels = train_set.getLabels()
                # List to store metrics per iteration
                mtdf_metrics_per_iter = []
                
                # -------------------------------------------------------------
                # Multilinear Model
                # -------------------------------------------------------------
                for J in FusedDim:
                    JFIX_RVARYING=[]
                    for r in range(len(Ranks)):
                        R=Ranks[r]
                        min_valid_loss = float('Inf')
                        curr_patience = patience
                        
                        # Initialize the model and the optimizer
                        model = NetworkModel(I, J, R, (batch_sz,lr,factor_lr,decay,scaler))
                        model.to(device)
                        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=decay)
                        class_loss = nn.CrossEntropyLoss() #BCEWithLogitsLoss
                        occ_loss = nn.CrossEntropyLoss()
                        Lambda_occ = 1.5
                        Lambda_class = 1.0
                        
                        train_loss_tdf = []
                        valid_loss_tdf = []
                        train_cm_tdf = []
                        valid_cm_tdf = []
                        
                        # start = time.time()
                        complete = True
                        i_best_model = 0
                        for e in range(epochs):
                            print ("\r\t\tIteration: " + str(nrep+1) + "/" + str(Nreps), " - J(" + str(J) + ") - R(" + str(R) + ") - Epoch " + str(e+1) + "/" + str(epochs) , end="")
                            model.train()
                            epoch_train_loss = 0.0
                            cm_train=[None,None]
                            cm_valid=[None,None]
                            for batch in train_iterator:
                                model.zero_grad()
                                
                                # Forward pass
                                X = batch[:nmodals]
                                X = [x.to(device) for x in X]
                                yclass = batch[-2].to(device)
                                yocc = batch[-1].to(device)
                                yclasshat,yocchat = model(X)
                                
                                # Compute the class loss
                                loss_1 = class_loss(yclasshat, yclass)
                                
                                # Compute the occ loss
                                loss_2 = occ_loss(yocchat, yocc)
                                
                                # Calcular la pérdida total sumando ambas pérdidas
                                loss = Lambda_class*loss_1 + Lambda_occ*loss_2
                                
                                # Clear all gradients
                                #optimizer.zero_grad()
                                
                                # Compute new gradients
                                loss.backward()
                                
                                # Update all weights
                                optimizer.step()
                                
                                # Calculate loss
                                epoch_train_loss += loss.item()
                                
                                # Get confusion matrices (classification)
                                y_true = yclass.cpu().data.numpy()
                                y_pred = np.round(yclasshat.cpu().data.numpy())
                                if cm_train[0] is None:
                                    cm_train[0]  = sk.metrics.confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), labels=list(labels[0]))
                                else:
                                    cm_train[0] += sk.metrics.confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), labels=list(labels[0]))
                                # Get confusion matrices (occlusion)
                                y_true = yocc.cpu().data.numpy()
                                y_pred = np.round(yocchat.cpu().data.numpy())
                                if cm_train[1] is None:
                                    cm_train[1]  = sk.metrics.confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), labels=list(labels[1]))
                                else:
                                    cm_train[1] += sk.metrics.confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), labels=list(labels[1]))
                            epoch_train_loss = epoch_train_loss / len(train_iterator)
                            
                            
                            # Model Validation ------------------------------------------------
                            model.eval()
                            avg_valid_loss = 0.0
                            with torch.no_grad():
                                # Terminate the training process if run into NaN
                                if np.isnan(epoch_train_loss):
                                    print("\n\tTraining got into NaN values...\n\n")
                                    complete = False
                                    break
                                
                                # VALIDATION
                                for batch in valid_iterator:
                                    # Forward pass
                                    X = batch[:nmodals]
                                    X = [x.to(device) for x in X]
                                    yclass = batch[-2].to(device)
                                    yocc = batch[-1].to(device)
                                    yclasshat,yocchat = model(X)
                                    
                                    # Compute the class loss
                                    loss_1 = class_loss(yclasshat, yclass)
                                    
                                    # Compute the occ loss
                                    loss_2 = occ_loss(yocchat, yocc)
                                    
                                    # Calcular la pérdida total sumando ambas pérdidas
                                    valid_loss = Lambda_class*loss_1 + Lambda_occ*loss_2
                                    avg_valid_loss += valid_loss.item()
                                    
                                    # Get confusion matrices (classification)
                                    y_true = yclass.cpu().data.numpy()
                                    y_pred = np.round(yclasshat.cpu().data.numpy())
                                    if cm_valid[0] is None:
                                        cm_valid[0]  = sk.metrics.confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), labels=list(labels[0]))
                                    else:
                                        cm_valid[0] += sk.metrics.confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), labels=list(labels[0]))
                                    # Get confusion matrices (occlusion)
                                    y_true = yocc.cpu().data.numpy()
                                    y_pred = np.round(yocchat.cpu().data.numpy())
                                    if cm_valid[1] is None:
                                        cm_valid[1]  = sk.metrics.confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), labels=list(labels[1]))
                                    else:
                                        cm_valid[1] += sk.metrics.confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), labels=list(labels[1]))
                                avg_valid_loss = avg_valid_loss / len(valid_iterator)
                                
                                
                                if np.isnan(avg_valid_loss):
                                    print("\n\tTraining got into NaN values...\n\n")
                                    complete = False
                                    break
                                
                            # END OF WITH.NO_GRAD() -----------------------------------
                            valid_loss_tdf.append(avg_valid_loss)
                            train_loss_tdf.append(epoch_train_loss)
                            valid_cm_tdf.append(cm_valid)
                            train_cm_tdf.append(cm_train)
                            
                            # PATIENCE SCHEME -----------------------------------------
                            if (avg_valid_loss < min_valid_loss):
                                curr_patience = patience
                                min_valid_loss = avg_valid_loss
                                torch.save(model, dirpath + "/" + fpath[1][4:-4] + "-MTDF-R" + str(R) + ".pth")
                                i_best_model = e
                            else:
                                curr_patience -= 1
                            
                            if curr_patience <= 0:
                                break
                        
                        
                        # Model Testing -------------------------------------------------------
                        # end = time.time()
                        if complete:
                            model = torch.load(dirpath + "/" + fpath[1][4:-4] + "-MTDF-R" + str(R) + ".pth")
                            model.to(device)
                            model.eval()
                            test_loss = 0.0
                            cm_class=np.zeros((6,6))
                            cm_occ=np.zeros((2,2))
                            for batch in test_iterator:
                                X = batch[:nmodals]
                                X = [x.to(device) for x in X]
                                yclass = batch[-2].to(device)
                                yocc = batch[-1].to(device)
                                yclasshat,yocchat = model(X)
                                
                                # Compute the class loss
                                loss_1 = class_loss(yclasshat, yclass)
                                
                                # Compute the occ loss
                                loss_2 = occ_loss(yocchat, yocc)
                                
                                # Calcular la pérdida total sumando ambas pérdidas
                                loss = Lambda_class*loss_1 + Lambda_occ*loss_2
                                test_loss += loss.item()
                                
                                # Evaluate
                                y_true_occ = yocc.cpu().data.numpy()
                                y_true_class = yclass.cpu().data.numpy()
                                y_pred_occ = np.round(yocchat.cpu().data.numpy())
                                y_pred_class = np.round(yclasshat.cpu().data.numpy())
                                cm_class  += sk.metrics.confusion_matrix(y_true_class.argmax(axis=1), y_pred_class.argmax(axis=1), labels=list(labels[0]))
                                cm_occ  += sk.metrics.confusion_matrix(y_true_occ.argmax(axis=1), y_pred_occ.argmax(axis=1), labels=list(labels[1]))
                                #m,delta,_= metrics_imbalance(cm,False)
                            test_loss /= len(test_iterator)
                            JFIX_RVARYING.append([train_loss_tdf, valid_loss_tdf, test_loss, train_cm_tdf, valid_cm_tdf,  [cm_class, cm_occ], i_best_model])
                        else:
                            print("\n\t\tTraining.Success: FAIL")
                            JFIX_RVARYING.append(None)
                    # END OF for r in range(len(Ranks))
                    mtdf_metrics_per_iter.append(JFIX_RVARYING)
                mtdf_metrics.append(mtdf_metrics_per_iter)
                
            # END OF for nrep in range(Nreps): ------------------------------------
            # Save metrics per iteration list
            end = time.time()
            print("\n\t\tElapsedTime: ", end - start, " (s)")
            print("-----------------------------\n\n")
            save_object(mtdf_metrics, dirpath + "/" + fpath[1][0:-4] + "-TDF" + ".pkl")
    except KeyboardInterrupt:
        print("\n\n#KeyboardInterrupt exception")
        print("Current metrics will be saved")
        save_object(mtdf_metrics, dirpath + "/" + fpath[1][0:-4] + "-TDF" + ".pkl")
        
        
    # END OF for ivideo in range(len(filenames)):
    
if __name__ == "__main__":
    main()
    
    