''' Applying Adversarial Auto-encoder for Estimating Human Walking Gait Index
    BSD 2-Clause "Simplified" License
    Author: Trong-Nguyen Nguyen'''

import argparse, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import *
from ops import *

### https://github.com/rothk/Stabilizing_GANs
def Discriminator_Regularizer(D1_logits, D1_arg, D2_logits, D2_arg):
    D1 = tf.nn.sigmoid(D1_logits)
    D2 = tf.nn.sigmoid(D2_logits)
    grad_D1_logits = tf.gradients(D1_logits, D1_arg)[0]
    grad_D2_logits = tf.gradients(D2_logits, D2_arg)[0]
    grad_D1_logits_norm = tf.norm(tf.reshape(grad_D1_logits, [mb_size,-1]), axis=1, keepdims=True)
    grad_D2_logits_norm = tf.norm(tf.reshape(grad_D2_logits, [mb_size,-1]), axis=1, keepdims=True)

    print(grad_D1_logits_norm.shape)
    print(D1.shape)

    reg_D1 = tf.multiply(tf.square(1.0-D1), tf.square(grad_D1_logits_norm))
    reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D2_logits_norm))
    disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)
    return disc_regularizer

''' constant parameters '''
mb_size = 512
z_dim = 16
X_dim = 256
h_dim = 96
lr = 1e-3
segment_lengths = np.array([1, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 240, 300, 400, 600, 1200])

tf.set_random_seed(1989)
np.random.seed(1989)

def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(16, 16), cmap='Greys_r')
    return fig

def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = tf.sqrt(6. / (in_dim + out_dim))
    return tf.random_normal(shape=size, stddev=xavier_stddev)

""" Q(z|X) """
X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2 = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2 = tf.Variable(tf.zeros(shape=[z_dim]))

theta_Q = [Q_W1, Q_W2, Q_b1, Q_b2]

def Q(X):
    h = lrelu(tf.matmul(X, Q_W1) + Q_b1, leak = 0.1, name = 'h_Q')
    z = tf.matmul(h, Q_W2) + Q_b2
    return z

""" P(X|z) """
P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_P = [P_W1, P_W2, P_b1, P_b2]

def P(z):
    h = lrelu(tf.matmul(z, P_W1) + P_b1, leak = 0.1, name = 'h_P')
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits

""" D(z) """
D_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

def D(z):
    h = lrelu(tf.matmul(z, D_W1) + D_b1, leak = 0.1, name = 'h_D')
    logits = tf.matmul(h, D_W2) + D_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits

""" Losses and solvers """
z_sample = Q(X)
recon_X, logits = P(z_sample)

# Sample from random z
X_samples, _ = P(z)

D_real, D_real_logits = D(z)
D_fake, D_fake_logits = D(z_sample)

gamma = tf.placeholder_with_default(2.0, shape=()) #for annelling
d_reg = Discriminator_Regularizer(D_real_logits, z, D_fake_logits, z_sample)
D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake)) + (gamma/2.0)*d_reg
G_loss = -tf.reduce_mean(tf.log(D_fake))
recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X))

AE_solver = tf.train.AdamOptimizer(lr).minimize(recon_loss, var_list=theta_P + theta_Q)
D_solver = tf.train.GradientDescentOptimizer(lr).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=theta_Q)

''' suggested epoch ranges '''
split_epochs = np.array([200,300])
l1o_epochs = np.array([[100,200],[100,200],[120,220],[100,200],[80,180],[100,200],[100,200],[100,200],[95,195]])

def main(argv):
    '''usage: python3 main.py -l -1 -e1 200 -e2 300 -o 0 -s 1 -f results.csv'''
    parser = argparse.ArgumentParser(description = 'cylindrical histogram AAE')
    parser.add_argument('-l', '--l1o', help = 'leave-one-out', default = -1)
    parser.add_argument('-e1', '--epoch1', help = 'first epoch for evaluation', default = -1)
    parser.add_argument('-e2', '--epoch2', help = 'last epoch for evaluation', default = -1)
    parser.add_argument('-o', '--overlap', help = 'use overlapping segments (sliding window)', default = 0)
    parser.add_argument('-s', '--sampling', help = 'save sampled histograms', default = 1)
    parser.add_argument('-f', '--file', help = 'file saving AUC results', default = None)
    args = vars(parser.parse_args())
    l1o = int(args['l1o'])
    epoch_start = int(args['epoch1'])
    epoch_end = int(args['epoch2'])
    overlapping = bool(int(args['overlap']))
    save_samples = bool(int(args['sampling']))
    result_file = args['file']
    ''' load histogram data '''
    loaded = np.load('dataset/DIRO_normalized_hists.npz')
    data = loaded['data']
    n_subject, n_gait, n_frame = data.shape[:3]
    if epoch_start == -1 and epoch_end == -1:
        if l1o < 0 or l1o >= n_subject:
            epoch_start, epoch_end = split_epochs
        else:
            epoch_start, epoch_end = l1o_epochs[l1o]
    assert epoch_start >= 0 and epoch_end >= epoch_start
    n_epoch = epoch_end + 10
    if l1o < 0 or l1o >= n_subject:
        separation = loaded['split']
        training_subjects = np.where(separation == 'train')[0]
        test_subjects = np.where(separation == 'test')[0]
    else:
        test_subjects = np.array([l1o])
        training_subjects = np.setdiff1d(list(range(n_subject)), test_subjects)
        
    print('training subjects: ' + str(training_subjects))

    training_img_normal = data[training_subjects, 0]
    test_img_normal = data[test_subjects, 0]
    test_img_abnormal = data[test_subjects, 1:]

    '''flatten data to 2D matrix'''
    training_img_normal = training_img_normal.reshape((-1,256))
    test_img_normal = test_img_normal.reshape((-1,256))
    test_img_abnormal = test_img_abnormal.reshape((-1,256))

    print('data shape:')
    print(training_img_normal.shape)
    print(test_img_normal.shape)
    print(test_img_abnormal.shape)
    print('')

    ''' AUC variables with size of n_seg_len * n_considered_epoch'''
    results_prob, results_disc, results_dist, results_dist_prob, results_dist_disc, results_full = \
        [np.zeros((len(segment_lengths), epoch_end - epoch_start + 1)) for i in range(6)]
    training_losses = np.zeros((n_epoch, 3))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # default values for annealing
        alpha = 0.005
        gamma0 = 2.0
        T = n_epoch * int(np.ceil(training_img_normal.shape[0] / mb_size))
        t = 0
        for epoch in range(n_epoch):

            indices = get_batch(training_img_normal.shape[0], mb_size)
            tmp_losses = np.zeros(3)

            for it in range(indices.shape[0]):
                t += 1

                X_mb = training_img_normal[indices[it]]
                z_mb = np.random.randn(mb_size, z_dim)

                _, recon_loss_curr = sess.run([AE_solver, recon_loss], feed_dict={X: X_mb, gamma: gamma0 * alpha**(t/T)})
                _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, z: z_mb})
                _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X: X_mb})

                tmp_losses += np.array([D_loss_curr, G_loss_curr, recon_loss_curr])

            tmp_losses /= indices.shape[0]
            training_losses[epoch,:] = tmp_losses

            print('Epoch %3d: D_loss %.3f, G_loss %.3f, Recon_loss: %.3f' % (epoch + 1, tmp_losses[0], tmp_losses[1], tmp_losses[2]))

            z_generated = np.random.randn(64, z_dim)
            if save_samples and (epoch + 1) % 50 == 0:
                if not os.path.exists('sampling/'):
                    os.makedirs('sampling/')
                samples = sess.run(X_samples, feed_dict={z: z_generated})
                fig = plot(samples)
                plt.savefig('sampling/{}.png'.format(str(epoch+1).zfill(3)), bbox_inches='tight')
                plt.close(fig)
            
            if (epoch+1) < epoch_start or (epoch+1) > epoch_end:
                continue

            result_idx = epoch + 1 - epoch_start

            max_exp_log = np.exp(log_likelihood(z_dim, np.zeros((1, z_dim))))
            assert len(max_exp_log) == 1
            max_exp_log = max_exp_log[0]
            
            '''assessment by Prob(z, Gaussian)'''
            z_abnormal = z_sample.eval({X: test_img_abnormal})
            z_normal = z_sample.eval({X: test_img_normal})
            prob_abnormal = (max_exp_log - np.exp(log_likelihood(z_dim, z_abnormal)))/max_exp_log
            assert np.amax(prob_abnormal) <= 1.0 and np.amin(prob_abnormal) >= 0.0
            prob_normal = (max_exp_log - np.exp(log_likelihood(z_dim, z_normal)))/max_exp_log
            assert np.amax(prob_normal) <= 1.0 and np.amin(prob_normal) >= 0.0
            re = assessment_full(prob_abnormal, prob_normal, segment_lengths, calc_mean = True, overlapping = overlapping)
            results_prob[:,result_idx] = re

            '''assessment by D(z|X)'''
            disc_abnormal = D_fake.eval({X: test_img_abnormal}).reshape(-1)
            disc_normal = D_fake.eval({X: test_img_normal}).reshape(-1)
            re = assessment_full(disc_abnormal, disc_normal, segment_lengths, calc_mean = True, overlapping = overlapping)
            results_disc[:,result_idx] = re

            '''assessment by Dist(X, X_hat)'''
            X_hat_abnormal = recon_X.eval({X: test_img_abnormal})
            X_hat_normal = recon_X.eval({X: test_img_normal})
            diff_abnormal = (np.mean((X_hat_abnormal - test_img_abnormal)**2, axis = 1))**0.5
            diff_normal = (np.mean((X_hat_normal - test_img_normal)**2, axis = 1))**0.5
            re = assessment_full(diff_abnormal, diff_normal, segment_lengths, calc_mean = True, overlapping = overlapping)
            results_dist[:,result_idx] = re

            '''use reconstruction error and weight'''
            ### weight calculation ###
            X_hat_train = recon_X.eval({X: training_img_normal})
            prob_train = (max_exp_log - np.exp(log_likelihood(z_dim, z_sample.eval({X: training_img_normal}))))/max_exp_log
            assert np.amax(prob_train) <= 1.0 and np.amin(prob_train) >= 0.0
            diff_train = (np.mean((X_hat_train - training_img_normal)**2, axis = 1))**0.5
            disc_train = D_fake.eval({X: training_img_normal}).reshape(-1)

            p = 0.125
            prob_normal, prob_abnormal, prob_train = prob_normal**p, prob_abnormal**p, prob_train**p

            ### (option) weighted sum: prob + dist ###
            W_diff1, W_prob1 = weight_calc2(diff_train, prob_train)
            seq_abnormal = W_diff1 * diff_abnormal - W_prob1 * prob_abnormal
            seq_normal = W_diff1 * diff_normal - W_prob1 * prob_normal
            re = assessment_full(seq_abnormal, seq_normal, segment_lengths, calc_mean = True, overlapping = overlapping)
            results_dist_prob[:,result_idx] = re

            ### (option) weighted sum: disc + dist ###
            W_diff2, W_disc2 = weight_calc2(diff_train, disc_train)
            seq_abnormal = W_diff2 * diff_abnormal + W_disc2 * disc_abnormal
            seq_normal = W_diff2 * diff_normal + W_disc2 * disc_normal
            re = assessment_full(seq_abnormal, seq_normal, segment_lengths, calc_mean = True, overlapping = overlapping)
            results_dist_disc[:,result_idx] = re

            ### (option) weighted sum ###
            W_prob, W_diff, W_disc = weight_calc3(prob_train, diff_train, disc_train)
            seq_abnormal = W_diff * diff_abnormal - W_prob * prob_abnormal + W_disc * disc_abnormal
            seq_normal = W_diff * diff_normal - W_prob * prob_normal + W_disc * disc_normal
            re = assessment_full(seq_abnormal, seq_normal, segment_lengths, calc_mean = True, overlapping = overlapping)
            results_full[:,result_idx] = re

    print('\nFINAL RESULTS (AVERAGE)')
    prob = show_results(results_prob, segment_lengths, str_title = '\nResults probability', return_mean = True)
    disc = show_results(results_disc, segment_lengths, str_title = '\nResults discriminator', return_mean = True)
    dist = show_results(results_dist, segment_lengths, str_title = '\nResults reconstruction', return_mean = True)
    dist_prob = show_results(results_dist_prob, segment_lengths, str_title = '\nResults dist + prob', return_mean = True)
    dist_disc = show_results(results_dist_disc, segment_lengths, str_title = '\nResults dist + disc', return_mean = True)
    full = show_results(results_full, segment_lengths, str_title = '\nResults combination', return_mean = True)
    plot_training_losses(training_losses, epoch_start, epoch_end)
    
    # save data to file
    if result_file:
        test_subjects_id = np.sum(np.array([test_subjects[i] * 10**(len(test_subjects)-1-i) for i in range(len(test_subjects))]))
        test_subjects_id *= np.ones(len(segment_lengths))
        data_to_save = np.transpose(np.vstack((test_subjects_id, segment_lengths, prob[:,0], disc[:,0], dist[:,0], dist_prob[:,0], dist_disc[:,0], full[:,0])))
        if os.path.isfile(result_file):
            loaded_data = np.loadtxt(result_file, delimiter = ',')
            data_to_save = np.concatenate((loaded_data, data_to_save), axis = 0)
        np.savetxt(result_file, data_to_save, delimiter = ',')
        print('results saved!')

    # plot AUC with different segment lengths
    lw = 1.7
    plt.figure(2)
    plt.plot(segment_lengths, prob[:,0], color = 'r', marker = '.', linewidth = lw, label = 'prob')
    plt.plot(segment_lengths, disc[:,0], color = 'g', marker = '.', linewidth = lw, label = 'disc')
    plt.plot(segment_lengths, dist[:,0], color = 'b', marker = '.', linewidth = lw, label = 'dist')
    plt.plot(segment_lengths, dist_prob[:,0], color = 'c', marker = '.', linewidth = lw, label = 'dist_prob')
    plt.plot(segment_lengths, dist_disc[:,0], color = 'm', marker = '.', linewidth = lw, label = 'dist_disc')
    plt.plot(segment_lengths, full[:,0], color = 'y', marker = '.', linewidth = lw, label = 'full')
    plt.xlabel('segment length')
    plt.ylabel('AUC')
    plt.legend(loc = 'upper right')
    plt.show()
    
if __name__ == '__main__':
    main(sys.argv)
