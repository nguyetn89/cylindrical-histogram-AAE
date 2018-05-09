import numpy as np
import random
import tensorflow as tf

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def window_separate(values, seg_len):
    assert seg_len <= len(values)
    return np.vstack(values[i:i+seg_len] for i in range(len(values)-seg_len+1))

def assessment(abnormal_values, normal_values, seg_len, calc_mean = True, overlapping = False):
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
    return auc

def assessment_full(prob_list_abnormal, prob_list_normal, seg_lens, calc_mean = True, overlapping = False, show_result = False):
    if show_result:
        print('abnormal sample: %d, normal sample: %d' % (prob_list_abnormal.size, prob_list_normal.size))
    if isinstance(seg_lens, int):
        seg_lens = [seg_lens]
    results = np.zeros(len(seg_lens))
    for i in range(len(seg_lens)):
        auc = assessment(prob_list_abnormal, prob_list_normal, seg_lens[i], calc_mean = calc_mean, overlapping = overlapping)
        results[i] = auc
        if show_result:
            print("(length %4d) auc = %.3f" % (seg_lens[i], auc))
    return results

def weight_calc3(latent_prob_seq, dist_seq, disc_seq, use_mean = True):
    m_latent = np.mean(latent_prob_seq) if use_mean else np.median(latent_prob_seq)
    m_dist = np.mean(dist_seq) if use_mean else np.median(dist_seq)
    m_disc = np.mean(disc_seq) if use_mean else np.median(disc_seq)
    w_latent = (m_latent + m_dist + m_disc) / m_latent
    w_dist = (m_latent + m_dist + m_disc) / m_dist
    w_disc = (m_latent + m_dist + m_disc) / m_disc
    return w_latent, w_dist, w_disc

def weight_calc2(dist_seq, other_seq, use_mean = True):
    m_dist = np.mean(dist_seq) if use_mean else np.median(dist_seq)
    m_other = np.mean(other_seq) if use_mean else np.median(other_seq)
    w_dist = (m_other + m_dist) / m_dist
    w_other = (m_other + m_dist) / m_other
    return w_dist, w_other

def write_results_to_file(filename, caption, data):
    with open(filename, "a") as myfile:
        myfile.write(caption + ' = [')
        for val in data:
            myfile.write(str(val))
            myfile.write(' ')
        myfile.write(']\n')

def log_likelihood(k, data): # Wikipedia
    mean = np.zeros((1,k))
    cov = np.identity(k)
    def calc_loglikelihood(residuals):
        return -0.5 * (np.log(np.linalg.det(cov)) + residuals.T.dot(np.linalg.inv(cov)).dot(residuals) \
                         + k * np.log(2 * np.pi))
    residuals = (data - mean)
    loglikelihood = np.apply_along_axis(calc_loglikelihood, 1, residuals)
    return loglikelihood

def get_batch(n_length, batch_size):
    idx_list = list(range(n_length))
    np.random.shuffle(idx_list)
    r = n_length % batch_size
    if r > 0:
        idx_list = np.concatenate((idx_list, np.random.randint(0, high = n_length, size = batch_size - r)), axis = 0)
    return idx_list.reshape((-1, batch_size))

def show_results(results, seg_len, str_title = '', return_mean = False):
    print(str_title)
    if isinstance(seg_len, int):
        seg_len = [seg_len]
    assert results.shape[0] == len(seg_len)
    if return_mean:
        ret = np.zeros((len(seg_len), 2))
    for i in range(len(seg_len)):
        tmp = results[i,:]
        tmp_mean = np.mean(tmp)
        tmp_std = np.std(tmp)
        if return_mean:
            ret[i] = np.array([tmp_mean, tmp_std])
        print('(%4d) AUC = %.4f (+%.4f)' % (seg_len[i], tmp_mean, tmp_std))
    if return_mean:
        return ret

def plot_training_losses(training_losses, epoch_start = 400, epoch_end = 500):
    X = list(range(training_losses.shape[0]))
    plt.figure()
    plt.plot(X, training_losses[:,0], color = 'r', linewidth = 2.5, label = 'D_loss')
    plt.plot(X, training_losses[:,1], color = 'g', linewidth = 2.5, label = 'G_loss')
    plt.plot(X, training_losses[:,2], color = 'b', linewidth = 2.5, label = 'recon_loss')
    plt.axvline(x = epoch_start)
    plt.axvline(x = epoch_end)
    plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)

