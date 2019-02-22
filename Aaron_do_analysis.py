from __future__ import print_function 
from __future__ import division
import numpy as np
import pandas as pd
import os
import copy
import random

NUM_CLASSES = 7

def get_sequentials():
    
    models = []
    '''
    for which in ["top","bot"]:
        for i in range(1,6):
            curr_ref = "seq/{}_{}".format(which,i)
            models.append(curr_ref)
    '''
    for i in range(1,101):
        curr_ref = "seq/seq_{}".format(i)
        models.append(curr_ref)
    
    return models

def run():
    
    val_or_test = "val"
    
    label_file = "./aaron_saved_arrays/{}_labels.npy".format(val_or_test)
    labels = np.load(label_file)
    logit_file = "./aaron_saved_arrays/{}_normed_logits.npy".format(val_or_test)
    logits = np.load(logit_file)
    
    total_preds = len(labels)
    
    models = get_sequentials()
    '''models = [
        "overall_naive",
        "select_naive"
        ]'''
    '''models = [
        "modified_regression"
        ]'''
    '''models = [
        "just_dark",
        "just_lime",
        "just_msr",
        "just_bimef",
        "just_ying",
        ]'''
    '''models = [
        "overall_naive",
        "select_naive",
        "acc_model",
        "acc_squared_model",
        "acc_fifth_model",
        "prec_model",
        "recall_model",
        "f1_model",
        "f1_squared_model",
        "f1_fifth_model",
        "f1_exp_model"
        ]'''
    

    excel_save = np.zeros((len(models),3*8))
    
    for model_num in range(len(models)):
        
        accuracy = 0
        cf_mat = np.zeros((NUM_CLASSES,3))
        
        model_file = "./aaron_saved_arrays/ens_models/{}.npy".format(models[model_num])
        curr_model = np.load(model_file)
        
        for i in range(total_preds):
            
            curr_pred = np.argmax(np.sum(np.multiply(logits[i],curr_model),axis=0))
            
            if curr_pred == labels[i]:
                cf_mat[curr_pred,0] += 1
                accuracy += 1/total_preds
                
            else:
                cf_mat[curr_pred,1] += 1
                cf_mat[labels[i],2] += 1
                
        out_mat = np.zeros((8,3))
        out_mat[7,:] = accuracy
        print("Accuracy for model {} is {}".format(models[model_num],accuracy))
        
        for i in range(NUM_CLASSES):
            
            precision = cf_mat[i,0]/(cf_mat[i,0] + cf_mat[i,1])
            recall = cf_mat[i,0]/(cf_mat[i,0] + cf_mat[i,2])
            f1score = 2*precision*recall/(precision+recall)
            
            print("     For class {}, precision is {}".format(i,precision))
            print("     For class {}, recall is {}".format(i,recall))
            print("     For class {}, f1score is {}".format(i,f1score))
            
            out_mat[i] = [precision,recall,f1score]
        
        excel_save[model_num] = out_mat.flatten()
        
        #metrics_file = "aaron_saved_arrays/metric_data/{}_{}.npy".format(val_or_test,models[model_num])
        #np.save(metrics_file,out_mat)
    
    excel_save = np.around(excel_save, decimals=4)
    df = pd.DataFrame(excel_save)
    print(df)
    df.to_excel("aaron_saved_arrays/last_model.xlsx", index=False, header=False)

if __name__ == "__main__":
    run()