from __future__ import print_function 
from __future__ import division
import numpy as np
import os
import copy
import random

def parse_log_file(f):
    
    #I'm splitting logits and labels to make things a bit easier later on
    logits = []
    labels = []
    print("Parsing: ",f)
    logfile = open(f,"r")
    for line in logfile:
		# Parse line into list of floats	
        parts = [float(x) for x in line.rstrip().split(",")]
        # Extract logits into their own list
        logits.append([x for x in parts[2:]])
        # Add entry into lines
        labels.append(int(parts[0]))
        
    logfile.close()
    return labels, logits
	

def run():
	
    print("starting")
    
    models = ["dark","lime","multiscaleRetinex","BIMEF","ying"]
    seeds = [12345,56789,63751]
	
    list_logits = "not_set"
    val_or_test = "val"

    
    for model in models:
	
        curr_logits = "not_set"
		
        for seed in seeds:
		
            #Get data from each of the logit files.  We're combining the data as we want it all evaluated togethor
            curr_dir = "./model_predictions/{}/seed{}/resnet_{}_100epoch_seed{}_testlogits.txt".format(val_or_test,seed,model,seed)
            if curr_logits == "not_set":
                curr_labels, curr_logits = parse_log_file(curr_dir)
            else:
                add_labels, add_logits = parse_log_file(curr_dir)
                curr_labels = np.concatenate((curr_labels,add_labels),axis=0)
                curr_logits = np.concatenate((curr_logits,add_logits),axis=0)
        
        if list_logits == "not_set":
            labels = curr_labels
            list_logits = [curr_logits]
        else:
            #Check for match between models
            assert(len(labels) == len(curr_labels))
            for i in range(len(labels)):
                assert( labels[i] == curr_labels[i] )
            
            #Add logits to singular 3D array
            list_logits.append(curr_logits)
        
        all_logits = np.stack(list_logits, axis = 1)
        
    print(all_logits.shape)
    for i in range(len(all_logits)):
        for j in range(5):
            all_logits[i,j] = np.exp(all_logits[i,j])/np.sum(np.exp(all_logits[i,j]))
    
    #Save logits and labels
    logit_file = "Aaron_saved_arrays/{}_normed_logits.npy".format(val_or_test)
    np.save(logit_file,all_logits)
    label_file = "Aaron_saved_arrays/{}_labels.npy".format(val_or_test)
    np.save(label_file,labels)
    
    print("done")


if __name__ == "__main__":
    run()