import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def window_separate(values, seg_len):
    assert seg_len <= len(values)
    return np.vstack(values[i:i+seg_len] for i in range(len(values)-seg_len+1))

def assessment(abnormal_values, normal_values, seg_len, calc_mean = False, overlapping = False):
    if calc_mean:
        if overlapping:
            abnormal_values = np.mean(window_separate(abnormal_values,seg_len), axis = 1)
            normal_values = np.mean(window_separate(normal_values,seg_len), axis = 1)
        else:
            abnormal_values = np.mean(np.split(abnormal_values,len(abnormal_values)//seg_len), axis = 1)
            normal_values = np.mean(np.split(normal_values,len(normal_values)//seg_len), axis = 1)
    else:
        if overlapping:
            abnormal_values = np.median(window_separate(abnormal_values,seg_len), axis = 1)
            normal_values = np.median(window_separate(normal_values,seg_len), axis = 1)
        else:
            abnormal_values = np.median(np.split(abnormal_values,len(abnormal_values)//seg_len), axis = 1)
            normal_values = np.median(np.split(normal_values,len(normal_values)//seg_len), axis = 1)

    labels = np.concatenate((np.ones(len(abnormal_values)),np.zeros(len(normal_values))), axis = 0)
    auc = roc_auc_score(labels, np.concatenate((abnormal_values, normal_values), axis=0))
    fpr, tpr, thresholds = roc_curve(labels, np.concatenate((abnormal_values, normal_values), axis=0), pos_label = 1)  #1 indicates label of positive class
    eer, thresh = EER_calc(fpr, tpr, thresholds)
    #
    TP = len(abnormal_values[abnormal_values>=thresh])
    FN = len(abnormal_values[abnormal_values<thresh])
    TN = len(normal_values[normal_values<thresh])
    FP = len(normal_values[normal_values>=thresh])
    sensitivity = TP/(TP+FN) if TP + FN > 0 else 0
    specificity = TN/(TN+FP) if TN + FP > 0 else 0
    precision = TP/(TP+FP) if TP + FP > 0 else 0
    accuracy = (TP+TN)/(TP+TN+FP+FN) if TP + TN + FP + FN > 0 else 0
    eer = 1 - accuracy
    F1 = 2*precision*sensitivity/(precision+sensitivity) if precision + sensitivity > 0 else 0
    return auc, eer, thresh, sensitivity, specificity, precision, accuracy, F1

'''return eer and threshold'''
def EER_calc(fpr, tpr, thresholds):
    if fpr[0] + tpr[0] > 0.0:
        fpr = np.insert(fpr, 0, 0.0)
        tpr = np.insert(tpr, 0, 0.0)
    if fpr[-1] + tpr[-1] < 2.0:
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)
    vt = np.array([-1, 1])
    p_found = np.array([None, None])
    thresh = None
    for i in range(fpr.size - 1):
        p1 = np.array([fpr[i], tpr[i]])
        p2 = np.array([fpr[i+1], tpr[i+1]])
        if np.sum(p2) == 1.0:
            p_found = p2
            thresh = thresholds[i]
            break
        
        p_intersect = point_intersect_ROC(p1, p2)
        v1 = (p1 - p_intersect)
        v2 = (p2 - p_intersect)
        #print('%d/%d' % (i, fpr.size - 1))
        #print(p_intersect)
        if np.sum(v1 * v2) < 0:
            p_found = p_intersect
            thresh = thresholds[i-1]
            break
        
    return p_found[0], thresh
    
'''find intersection between line (p1, p2) and line y = 1 - x'''
def point_intersect_ROC(p1, p2):
    if p1[0] == p2[0]:
        return np.array([p1[0], 1 - p1[0]])
    a = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - a * p1[0]
    px = (1 - b) / (a + 1)
    py = 1 - px
    return np.array([px, py])
