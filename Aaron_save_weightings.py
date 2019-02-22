from __future__ import print_function 
from __future__ import division
import numpy as np
import os
import copy
import random

NUM_MODELS = 5
NUM_CLASSES = 7


def save_naives(m_list):
        
    #Saving singular models
    for i in range(NUM_MODELS):
        
        curr_weights = np.zeros((NUM_MODELS,NUM_CLASSES))
        curr_weights[i] = np.ones(NUM_CLASSES)
        
        file_loc = "aaron_saved_arrays/ens_models/just_{}.npy".format(m_list[i])
        
        np.save(file_loc,curr_weights)
    
    #Saving naive overall model
    naive_file = "aaron_saved_arrays/ens_models/overall_naive.npy"
    np.save(naive_file,np.ones((NUM_MODELS,NUM_CLASSES))*(1/NUM_MODELS))
    
    #Input of some selection of naive models
    which_models = [0, 1, 1, 0, 1]
    curr_weights = np.zeros((NUM_MODELS,NUM_CLASSES))
    curr_weights[which_models] = np.ones(NUM_CLASSES)
    naive_select_file = "aaron_saved_arrays/ens_models/select_naive.npy"
    np.save(naive_select_file,curr_weights*(1/sum(which_models)))
    
    return
        

def save_metrics(m_list):
    
    acc_metric = np.zeros((NUM_MODELS,NUM_CLASSES))
    prec_metric = np.zeros((NUM_MODELS,NUM_CLASSES))
    recall_metric = np.zeros((NUM_MODELS,NUM_CLASSES))
    f1_metric = np.zeros((NUM_MODELS,NUM_CLASSES))
    
    for i in range(NUM_MODELS):
        #Pull metrics from current individual model data
        metric_file = "./aaron_saved_arrays/metric_data/val_just_{}.npy".format(m_list[i])
        curr_metric = np.load(metric_file)
        
        acc_metric[i,:] = curr_metric[7,0]
        prec_metric[i] = curr_metric[:7,0]
        recall_metric[i] = curr_metric[:7,1]
        f1_metric[i] = curr_metric[:7,2]
      
    #Perform squaring and fith powering math
    acc_squared_metric = np.power(acc_metric,2)
    acc_fifth_metric = np.power(acc_metric,5)
    f1_squared_metric = np.power(f1_metric,2)
    f1_fifth_metric = np.power(f1_metric,5)
    f1_exp_metric = np.exp(f1_metric)
    
    for i in range(NUM_CLASSES):
        #Normalize the columns (though it really shouldn't matter)
        acc_metric[:,i] = acc_metric[:,i]/sum(acc_metric[:,i])
        acc_squared_metric[:,i] = acc_squared_metric[:,i]/sum(acc_squared_metric[:,i])
        acc_fifth_metric[:,i] = acc_fifth_metric[:,i]/sum(acc_fifth_metric[:,i])
        prec_metric[:,i] = prec_metric[:,i]/sum(prec_metric[:,i])
        recall_metric[:,i] = recall_metric[:,i]/sum(recall_metric[:,i])
        f1_metric[:,i] = f1_metric[:,i]/sum(f1_metric[:,i])
        f1_squared_metric[:,i] = f1_squared_metric[:,i]/sum(f1_squared_metric[:,i])
        f1_exp_metric[:,i] = f1_exp_metric[:,i]/sum(f1_exp_metric[:,i])
    
    #Saves each model individually
    np.save("aaron_saved_arrays/ens_models/acc_model.npy",acc_metric)
    np.save("aaron_saved_arrays/ens_models/acc_squared_model.npy",acc_squared_metric)
    np.save("aaron_saved_arrays/ens_models/acc_fifth_model.npy",acc_fifth_metric)
    np.save("aaron_saved_arrays/ens_models/prec_model.npy",prec_metric)
    np.save("aaron_saved_arrays/ens_models/recall_model.npy",recall_metric)
    np.save("aaron_saved_arrays/ens_models/f1_model.npy",f1_metric)
    np.save("aaron_saved_arrays/ens_models/f1_squared_model.npy",f1_squared_metric)
    np.save("aaron_saved_arrays/ens_models/f1_fifth_model.npy",f1_fifth_metric)
    np.save("aaron_saved_arrays/ens_models/f1_exp_model.npy",f1_exp_metric)
    
    return
    
    
def save_regression():
    
    labels = np.load("./aaron_saved_arrays/val_labels.npy")
    logits = np.load("./aaron_saved_arrays/val_logits.npy")
    
    num_preds = len(labels)
    out_weights = np.zeros((NUM_MODELS,NUM_CLASSES))
    
    for i in range(NUM_CLASSES):
        
        curr_coeffs = np.zeros((num_preds,NUM_MODELS))
        curr_outcome = np.zeros(num_preds)
        
        for j in range(num_preds):
            
            curr_coeffs[j] = logits[j,:,i]
            if( labels[j] == i ):
                curr_outcome[j] = 1
            else:
                curr_outcome[j] = 0
        
        resulting_weights = np.linalg.lstsq(curr_coeffs, curr_outcome, rcond=None)[0]
        
        out_weights[:,i] = np.exp(resulting_weights)/sum(np.exp(resulting_weights))
    
    print(out_weights)    
    np.save("aaron_saved_arrays/ens_models/modified_regression.npy",out_weights)
    
    return
    
def sequential_models():
    
    #Top/Bottom models
    top_5 = [3,5,2,1,4]
    bot_5 = [4,1,2,5,3]
    
    curr_top = np.zeros((NUM_MODELS,NUM_CLASSES))
    curr_bot = np.zeros((NUM_MODELS,NUM_CLASSES))

    for i in range(5):
        
        curr_top[top_5[i]-1,:] = 1.
        curr_bot[bot_5[i]-1,:] = 1.
        
        file_name = "aaron_saved_arrays/ens_models/seq/top_{}.npy".format(i+1)
        np.save(file_name,curr_top*(1/(i+1)))
        file_name = "aaron_saved_arrays/ens_models/seq/bot_{}.npy".format(i+1)
        np.save(file_name,curr_bot*(1/(i+1)))
        
    for i in range(101):
        #MSR -> Ying
        curr_weights = np.zeros((NUM_MODELS,NUM_CLASSES))
        curr_weights[2] = np.ones(NUM_CLASSES)*(1-i*.01)
        curr_weights[4] = np.ones(NUM_CLASSES)*(i*.01)
        file_name = "aaron_saved_arrays/ens_models/seq/seq_{}.npy".format(i)
        np.save(file_name,curr_weights)

def run():
    
    model_list = ["dark","lime","msr","bimef","ying"]
    
    assert(NUM_MODELS == len(model_list))
    
    #save_naives(model_list)
    
    #save_metrics(model_list)
    
    sequential_models()
    
    #save_regression()





if __name__ == "__main__":
    run()