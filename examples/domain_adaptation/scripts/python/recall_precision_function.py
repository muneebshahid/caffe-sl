# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 17:15:00 2014

@author: Tayyab Naseer

"""

import argparse
import numpy as np
from os import listdir
from os.path import isfile, join
import copy



def best_matches(x):
	best_mat = np.zeros((query_size, db_size),dtype=np.uint8)
        
	
	threshold = x / float(100)
	for i, line in enumerate(score_matrix):
		max_index = line.argmax()
		if score_matrix[i, max_index] > threshold:
            		best_mat[i, max_index] = 1
			#result_indices[i,1] = max_index
		else:
			q=1
	return best_mat		

def make_path_matrix(results_file):

    print 'Making Path Matrix'	
    DB_SIZE = int(args.DB_SIZE)
    QRY_SIZE = int(args.QRY_SIZE)
    path_hypothesis = open(results_file, "r")

    queries=[]
    db=[]
    for columns in (raw.strip().split() for raw in path_hypothesis ):
        db.append(columns[1])
        queries.append(columns[0])

    

        

    queries=np.array(queries).astype(np.float32)
    db=np.array(db).astype(np.float32)


    result_mat = np.empty((QRY_SIZE,DB_SIZE))
    result_mat[:]=0

    for q in range(len(queries)):
    
        if db[q]==-1:
            continue
    
        result_mat[np.int32(queries[q]),np.int32(db[q])] = 1


    #res = remove_db_stoppages(result_mat)
    return result_mat

def calc_PR(gt_data,result_hp,match_accuracy, is_bm):
    
    
   
    
    
    if (is_bm==0):
        #hp_data1 = np.loadtxt(result_hp) 
        hp_data = make_path_matrix(result_hp)
    else:
        hp_data = result_hp   

    print 'GT data',gt_data.shape
    print 'Result Data',hp_data.shape
    assert np.shape(gt_data) == np.shape(hp_data)

    positives = np.sum(gt_data)
    t_pos = 0
    f_pos = 0
    f_neg = 0
    t_neg = 0
    
        

    for gt_match, hp_match in zip(gt_data, hp_data):
        gt_match = np.where(gt_match)[0]
        hp_match = np.where(hp_match)[0]

        l_gt = len(gt_match)
        l_hp = len(hp_match)

        diff = abs(l_gt - l_hp)

        if l_gt == 0 and l_hp == 0:
            t_neg += 1
            continue
        
        elif l_gt > l_hp :
	    		f_neg += diff
        prev_hp_element=-18 
        for hp_element in hp_match:
			if (not (hp_element - prev_hp_element > 20)):
 				prev_hp_element = hp_element
				continue
			prev_hp_element = hp_element
			match_found = False
	    		for gt_element in gt_match:
		
	    			if abs(hp_element - gt_element) <= match_accuracy:
					
	    				match_found = True
					
	    				break
			if match_found:
				t_pos += 1
			else:
				if(hp_element!=-1):
    	    				f_pos += 1
    
    if (t_pos + f_pos !=0):
    
	    print "t_pos: {0}, t_neg: {1}, f_pos: {2}, f_neg: {3}".format(t_pos, t_neg, f_pos, f_neg)
    	    print 'Precision %f'%(t_pos / float(t_pos + f_pos))
    
 	    print 'Recall %f'%(t_pos / float(positives))
    	    rec = (t_pos / float(positives))
            prec=(t_pos / float(t_pos + f_pos)) 
    else:
	rec,prec=0,0
    return rec,prec	 
    #precision.append(t_pos / float(t_pos + f_pos))
    
    #recall.append( t_pos / float(t_pos + f_neg))
    
        



parser = argparse.ArgumentParser()
parser.add_argument('path_gt', type=str, help='path_groundtruth_matrixfile')
parser.add_argument('score_file', type=str, help='SimilarityMatrix')
parser.add_argument('path_hp', type=str, help='path_hypothesis_dir')
parser.add_argument('match_accuracy', type=str, help='accuracy thresh')
parser.add_argument('cost_lower_lim', type=int, help='lower lim for cost')
parser.add_argument('cost_upper_lim', type=int, help='upper lim for cost')
parser.add_argument('cost_incr', type=int, help='cost_incr')
parser.add_argument('DB_SIZE', type=int, help='please enter the size of the database')
parser.add_argument('QRY_SIZE', type=int, help='please enter the size of the queries')
parser.add_argument('curve_name_path', type=str, help='pr_name_path')
parser.add_argument('curve_name_gbm', type=str, help='pr_name_gbm')

last_gt_element=-1
args = parser.parse_args()
path_gt = args.path_gt
gt_data = np.loadtxt(path_gt)


#gt_data = remove_db_stoppages(gt_)


query_size= args.QRY_SIZE
db_size=args.DB_SIZE


#try:
#    score_matrix = np.load(args.score_file+'.npy')
#    print 'Loaded Scores'
    
#except:
    
#    score_matrix = np.loadtxt(args.score_file+'.txt')
#    print 'Loaded Scores'





path_hp = args.path_hp

match_accuracy = int(args.match_accuracy)


if __name__ == "__main__":
   



    
    filenames= [f for f in listdir(path_hp) if isfile(join(path_hp,f))  ]
    filenames.sort()
    recall=[]
    precision=[]
    
    recall_bm=[]
    precision_bm=[]

    for files_ in filenames:
     
     
	[rec,prec] = calc_PR(gt_data,join(path_hp,files_),match_accuracy,0)
        recall.append(rec)
        precision.append(prec) 

    A = zip(recall,precision)
    np.savetxt(args.curve_name_path,A,fmt='%0.6f')
    
#    for cost in range(args.cost_lower_lim, args.cost_upper_lim+1, args.cost_incr):
#        bm_mat = best_matches(cost)
#        [rec1,prec1] = calc_PR(gt_data,bm_mat,match_accuracy,1)
#        recall_bm.append(rec1)
#        precision_bm.append(prec1)
    
#    B = zip(recall_bm,precision_bm)
#    np.savetxt(args.curve_name_gbm,B,fmt='%0.6f')
    
   
   


