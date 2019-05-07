#i Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)

""" Wasserstein Auto-Encoder models

"""
import math
import sys
import time
import os
import numpy as np
import tensorflow as tf
import logging
import ops
import utils
from models import encoder, decoder, z_adversary
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

import pdb

class WAE(object):

    def __init__(self, opts, sess):

        logging.error('Building the Tensorflow Graph')

        self.sess = sess
        self.opts = opts
        
        self.data_shape = opts['datashape']

        # Placeholders

        self.add_inputs_placeholders()
        self.add_training_placeholders()

        # Transformation ops

        # Encode the content of sample_points placeholder
        res = encoder(opts, inputs=self.sample_points,
                      is_training=self.is_training)
        self.enc_mean, self.enc_sigmas = None, None
        self.encoded, _ = res

        # Decode the points encoded above (i.e. reconstruct)
        self.reconstructed, self.reconstructed_logits = \
                decoder(opts, noise=self.encoded,
                        is_training=self.is_training)

        # Decode the content of sample_noise
        self.decoded, self.decoded_logits = \
            decoder(opts, reuse=True, noise=self.sample_noise,
                    is_training=self.is_training)
        
        # Main network
        with tf.variable_scope("main"):
            
            self.auxi_weights_list = self.t_generator(self.encoded, self.t_keep_prob)

            main_prev = self.sample_points
            for i in range(1, len(opts['main_info'])-1):
                if opts['auxi_info'][i-1]:
                    main_prev = ops.auxilinear(opts, main_prev, opts['main_info'][i], self.auxi_weights_list[i-1], scope='main_%d' % i)
                    main_prev = tf.nn.dropout(main_prev, self.main_keep_prob)
                    main_prev = tf.nn.relu(main_prev)
                else:
                    main_prev = ops.linear(opts, main_prev, opts['main_info'][i], scope='main_%d' % i)
                    main_prev = tf.nn.dropout(main_prev, self.main_keep_prob)
                    main_prev = tf.nn.relu(main_prev)
            
            if opts['auxi_info'][-1]:
                main_prev = ops.auxilinear(opts, main_prev, opts['main_info'][-1], self.auxi_weights_list[-1], scope='main_%d' % len(opts['auxi_info']))
            else:
                main_prev = ops.linear(opts, main_prev, opts['main_info'][-1], scope='main_%d' % len(opts['auxi_info']))
            
            self.prediction = main_prev
        
        
        # Objectives, losses, penalties
        self.former_placeholder()

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.sample_labels, logits=self.prediction))
        self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.sample_labels,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        self.penalty, self.loss_gan = self.matching_penalty()
        self.loss_reconstruct = self.reconstruction_loss(
            self.opts, self.sample_points, self.reconstructed)
        
        with tf.variable_scope("main"):
            self.trans_loss = self.transfer_loss()
        
        self.reg_loss = self.regularization_loss()
        self.f_loss = self.encoder_loss()
        
                        
        self.wae_loss = self.rec_lambda * self.loss_reconstruct + \
                         self.wae_lambda * self.penalty + \
                         self.reg_lambda * self.reg_loss + \
                         self.f_lambda * self.f_loss
        self.main_loss = self.main_lambda * self.cross_entropy + \
                         self.trans_lambda * self.trans_loss
                         

        # self.blurriness = self.compute_blurriness()

        if opts['e_pretrain']:
            self.loss_pretrain = self.pretrain_loss()
        else:
            self.loss_pretrain = None

        # self.add_least_gaussian2d_ops()

        # Optimizers, savers, initializer

        self.add_optimizers()
        self.add_savers()
        self.init = tf.global_variables_initializer()
        
    def t_generator(self, z, t_keep_prob, reuse=None):

        opts = self.opts

        t_weights_list = []

        for i in range(1, len(opts['main_info'])-1):
            if opts['auxi_info'][i-1]:
                auxi_weights = ops.t_network(opts, z, opts['auxi_info'][i-1] * opts['main_info'][i], i, t_keep_prob, reuse)
                auxi_weights = tf.reshape(auxi_weights, [-1, opts['auxi_info'][i-1], opts['main_info'][i]])
                t_weights_list.append(auxi_weights)
            else:
                t_weights_list.append(None)

        if opts['auxi_info'][-1]:
            #If there are auxiliary weights in the last layer
            auxi_weights = ops.t_network(opts, z, opts['auxi_info'][-1] * opts['main_info'][-1], len(opts['auxi_info']), t_keep_prob, reuse)
            auxi_weights = tf.reshape(auxi_weights, [-1, opts['auxi_info'][-1], opts['main_info'][-1]])
            t_weights_list.append(auxi_weights)
        else:
            t_weights_list.append(None)

        return t_weights_list

    def former_placeholder(self):
        
        # random z sampled from normal distribution
        self.random_z = tf.placeholder(tf.float32, [None, self.opts['zdim']])

        self.f_g_z = tf.placeholder(tf.float32, [None, self.opts['zdim']])

        # images generated from random z
        self.pseudo_G_z = tf.placeholder(tf.float32, [None] + self.opts['datashape'])
        # auxiliary weights T(z) generated from random z with old T-network
        self.pseudo_T_z = []
        # auxiliary weights T(z) generated from random z with new T-network
        # self.new_auxi_weights_list = []
        for l in range(len(self.opts['auxi_info'])):
            if self.opts['auxi_info'][l]:
                pseudo_auxi_weights = tf.placeholder(tf.float32, [None, self.opts['auxi_info'][l], self.opts['main_info'][l+1]])
                self.pseudo_T_z.append(pseudo_auxi_weights)
                # new_auxi_weights = tf.placeholder(tf.float32, [None, self.opts['auxi_info'][l], self.opts['main_info'][l+1]])
                # self.new_auxi_weights_list.append(new_auxi_weights)
            else:
                self.pseudo_T_z.append(tf.placeholder(tf.float32))
                # self.new_auxi_weights_list.append(tf.placeholder(tf.float32))
        # weights and biases in old main network
        self.w_old = []
        self.b_old = []
        for l in range(len(self.opts['auxi_info'])):
            if self.opts['main_info'][l] - self.opts['auxi_info'][l]:                
                weights_old = tf.placeholder(tf.float32, [self.opts['main_info'][l] - self.opts['auxi_info'][l], self.opts['main_info'][l+1]])
                self.w_old.append(weights_old)
            else:
                self.w_old.append(tf.placeholder(tf.float32))
            bias_old = tf.placeholder(tf.float32, self.opts['main_info'][l+1])
            self.b_old.append(bias_old)
                
    def former_init(self):
        
        random_z = np.zeros((self.opts['z_size'], self.opts['zdim']))
        
        f_g_z = np.zeros((self.opts['z_size'], self.opts['zdim']))

        pseudo_G_z = np.zeros(tuple([self.opts['z_size']] + self.opts['datashape']))
        pseudo_T_z = []
        for l in range(len(self.opts['auxi_info'])):
            if self.opts['auxi_info'][l]:
                pseudo_auxi_weights = np.zeros((self.opts['z_size'], self.opts['auxi_info'][l], self.opts['main_info'][l+1]))
                pseudo_T_z.append(pseudo_auxi_weights)
            else:
                pseudo_T_z.append(np.zeros(1))
        w_old = []
        b_old = []
        for l in range(len(self.opts['auxi_info'])): 
            if self.opts['main_info'][l] - self.opts['auxi_info'][l]:           
                weights_old = np.zeros((self.opts['main_info'][l] - self.opts['auxi_info'][l], self.opts['main_info'][l+1]))
                w_old.append(weights_old)
            else:
                w_old.append(None)
            bias_old = np.zeros(self.opts['main_info'][l+1])
            b_old.append(bias_old)
        
        pred = 1e-8 * np.ones((self.opts['z_size'], self.opts['main_info'][-1]))
        
        return random_z, pseudo_G_z, pseudo_T_z, w_old, b_old, f_g_z, pred

    def transfer_loss(self):
        
        loss = 0.0
        
        auxi_info = self.opts['auxi_info']
        main_info = self.opts['main_info']
        
        pseudo_G_z_tf = tf.cast(self.pseudo_G_z, tf.float32)
        
        pseudo_old = tf.reshape(pseudo_G_z_tf, [-1, np.prod(self.opts['datashape'])])
        pseudo_new = tf.reshape(pseudo_G_z_tf, [-1, np.prod(self.opts['datashape'])])

        self.new_auxi_weights_list = self.t_generator(self.f_g_z, t_keep_prob=1.0, reuse=True)
        
        for l in range(len(auxi_info)-1):

            if auxi_info[l]:
                new_auxi_weights = self.new_auxi_weights_list[l]
                old_auxi_weights = self.pseudo_T_z[l]
                
                if main_info[l] - auxi_info[l]:
                    pseudo_new_main = pseudo_new[:,:main_info[l] - auxi_info[l]]
                    pseudo_new_auxi = tf.expand_dims(pseudo_new[:,main_info[l] - auxi_info[l]:],1)
                    pseudo_old_main = pseudo_old[:,:main_info[l] - auxi_info[l]]
                    pseudo_old_auxi = tf.expand_dims(pseudo_old[:,main_info[l] - auxi_info[l]:],1)
                    
                    with tf.variable_scope('main_%d' % (l+1), reuse = True):
                        w_new = tf.get_variable('W')
                        b_new = tf.get_variable('b')
                    pseudo_new = tf.matmul(pseudo_new_main, w_new) + tf.squeeze(tf.matmul(pseudo_new_auxi, new_auxi_weights)) + b_new
                    pseudo_old = tf.matmul(pseudo_old_main, self.w_old[l]) + tf.squeeze(tf.matmul(pseudo_old_auxi, old_auxi_weights)) + self.b_old[l]
                
                else:
                    pseudo_new_auxi = tf.expand_dims(pseudo_new[:,main_info[l] - auxi_info[l]:],1)
                    pseudo_old_auxi = tf.expand_dims(pseudo_old[:,main_info[l] - auxi_info[l]:],1)

                    with tf.variable_scope('main_%d' % (l+1), reuse = True):
                        b_new = tf.get_variable('b')
                    pseudo_new = tf.squeeze(tf.matmul(pseudo_new_auxi, new_auxi_weights)) + b_new
                    pseudo_old = tf.squeeze(tf.matmul(pseudo_old_auxi, old_auxi_weights)) + self.b_old[l]          

                pseudo_new = tf.nn.relu(pseudo_new)
                pseudo_old = tf.nn.relu(pseudo_old)

                diff = pseudo_new - pseudo_old
                loss += tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis = [1]))
                
                # pseudo_new = tf.nn.relu(pseudo_new)
                # pseudo_old = tf.nn.relu(pseudo_old)
            else:
                
                with tf.variable_scope('main_%d' % (l+1), reuse = True):
                    w_new = tf.get_variable('W')
                    b_new = tf.get_variable('b')
                pseudo_new = tf.matmul(pseudo_new, w_new) + b_new
                pseudo_old = tf.matmul(pseudo_old, self.w_old[l]) + self.b_old[l]

                pseudo_new = tf.nn.relu(pseudo_new)
                pseudo_old = tf.nn.relu(pseudo_old)

                diff = pseudo_new - pseudo_old
                loss += tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis = [1]))

                # pseudo_new = tf.nn.relu(pseudo_new)
                # pseudo_old = tf.nn.relu(pseudo_old)
        
        if auxi_info[-1]:    
            new_auxi_weights = self.new_auxi_weights_list[-1]
            old_auxi_weights = self.pseudo_T_z[-1]
            
            if main_info[-2] - auxi_info[-1]:
                pseudo_new_main = pseudo_new[:,:main_info[-2] - auxi_info[-1]]
                pseudo_new_auxi = tf.expand_dims(pseudo_new[:,main_info[-2] - auxi_info[-1]:],1)
                pseudo_old_main = pseudo_old[:,:main_info[-2] - auxi_info[-1]]
                pseudo_old_auxi = tf.expand_dims(pseudo_old[:,main_info[-2] - auxi_info[-1]:],1)
                
                with tf.variable_scope('main_%d' % (len(auxi_info)), reuse = True):
                    w_new = tf.get_variable('W')
                    b_new = tf.get_variable('b')
                pseudo_new = tf.matmul(pseudo_new_main, w_new) + tf.squeeze(tf.matmul(pseudo_new_auxi, new_auxi_weights)) + b_new
                pseudo_old = tf.matmul(pseudo_old_main, self.w_old[-1]) + tf.squeeze(tf.matmul(pseudo_old_auxi, old_auxi_weights)) + self.b_old[-1]
            else:
                pseudo_new_auxi = tf.expand_dims(pseudo_new[:,main_info[-2] - auxi_info[-1]:],1)
                pseudo_old_auxi = tf.expand_dims(pseudo_old[:,main_info[-2] - auxi_info[-1]:],1)

                with tf.variable_scope('main_%d' % (len(auxi_info)), reuse = True):
                    b_new = tf.get_variable('b')
                pseudo_new = tf.squeeze(tf.matmul(pseudo_new_auxi, new_auxi_weights)) + b_new
                pseudo_old = tf.squeeze(tf.matmul(pseudo_old_auxi, old_auxi_weights)) + self.b_old[-1]


            diff = pseudo_new - pseudo_old
            loss += tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis = [1]))

        else:
            with tf.variable_scope('main_%d' % (len(auxi_info)), reuse = True):
                w_new = tf.get_variable('W')
                b_new = tf.get_variable('b')
            pseudo_new = tf.matmul(pseudo_new, w_new) + b_new
            pseudo_old = tf.matmul(pseudo_old, self.w_old[-1]) + self.b_old[-1]

            diff = pseudo_new - pseudo_old
            loss += tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis = [1]))
            
        return loss
    
    def regularization_loss(self):
        opts = self.opts
        
        self.random_z_recon, _ = encoder(opts, inputs=self.pseudo_G_z, reuse=True, is_training=self.is_training)
        loss_1 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.random_z_recon - self.random_z), axis = [1])))
        # tf.sqrt()
        self.G_new_z, _  = decoder(opts, noise=self.random_z, reuse=True, is_training=self.is_training)
    
        diff = tf.reshape(self.G_new_z, [-1, np.prod(self.opts['datashape'])]) -\
                 tf.reshape(self.pseudo_G_z, [-1, np.prod(self.opts['datashape'])])
        loss_2 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(diff), axis = [1])))
    
        loss = loss_1 + loss_2
        
        return loss

    def encoder_loss(self):

        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.random_z_recon - self.f_g_z), axis = [1])))

        return loss
    
    def add_inputs_placeholders(self):
        opts = self.opts
        shape = self.data_shape
        data = tf.placeholder(
            tf.float32, [None] + shape, name='real_points_ph')
        noise = tf.placeholder(
            tf.float32, [None] + [opts['zdim']], name='noise_ph')
        labels = tf.placeholder(
            tf.int32, [None] + [10], name='labels_ph')  

        self.sample_points = data
        self.sample_noise = noise
        self.sample_labels = labels

    def add_training_placeholders(self):

        decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        wae_lambda = tf.placeholder(tf.float32, name='wae_lambda_ph')
        rec_lambda = tf.placeholder(tf.float32, name='rec_lambda_ph')
        trans_lambda = tf.placeholder(tf.float32, name='trans_lambda_ph')
        reg_lambda = tf.placeholder(tf.float32, name='reg_lambda_ph')
        f_lambda = tf.placeholder(tf.float32, name='f_lambda_ph')
        main_lambda = tf.placeholder(tf.float32, name='main_lambda_ph')

        main_keep_prob = tf.placeholder(tf.float32, name='main_keep_prob_ph')
        t_keep_prob = tf.placeholder(tf.float32, name='t_keep_prob_ph')
        is_training = tf.placeholder(tf.bool, name='is_training_ph')
        false = tf.placeholder(tf.bool, name='false')
        
        self.lr_decay = decay
        self.wae_lambda = wae_lambda
        self.rec_lambda = rec_lambda
        self.trans_lambda = trans_lambda
        self.reg_lambda = reg_lambda
        self.f_lambda = f_lambda
        self.main_lambda = main_lambda

        self.is_training = is_training
        self.false = false
        self.main_keep_prob = main_keep_prob
        self.t_keep_prob = t_keep_prob

    def pretrain_loss(self):
        opts = self.opts
        # Adding ops to pretrain the encoder so that mean and covariance
        # of Qz will try to match those of Pz
        mean_pz = tf.reduce_mean(self.sample_noise, axis=0, keep_dims=True)
        mean_qz = tf.reduce_mean(self.encoded, axis=0, keep_dims=True)
        mean_loss = tf.reduce_mean(tf.square(mean_pz - mean_qz))
        cov_pz = tf.matmul(self.sample_noise - mean_pz,
                           self.sample_noise - mean_pz, transpose_a=True)
        cov_pz /= opts['e_pretrain_sample_size'] - 1.
        cov_qz = tf.matmul(self.encoded - mean_qz,
                           self.encoded - mean_qz, transpose_a=True)
        cov_qz /= opts['e_pretrain_sample_size'] - 1.
        cov_loss = tf.reduce_mean(tf.square(cov_pz - cov_qz))
        return mean_loss + cov_loss

    def add_savers(self):
        opts = self.opts
        saver = tf.train.Saver(max_to_keep=10)
        tf.add_to_collection('real_points_ph', self.sample_points)
        tf.add_to_collection('noise_ph', self.sample_noise)
        tf.add_to_collection('is_training_ph', self.is_training)
        if self.enc_mean is not None:
            tf.add_to_collection('encoder_mean', self.enc_mean)
            tf.add_to_collection('encoder_var', self.enc_sigmas)
        tf.add_to_collection('encoder', self.encoded)
        tf.add_to_collection('decoder', self.decoded)
        self.saver = saver


    def matching_penalty(self):
        loss_gan = None
        sample_qz = self.encoded
        sample_pz = self.sample_noise
        
        loss_match = self.mmd_penalty(sample_qz, sample_pz)

        return loss_match, loss_gan

    def mmd_penalty(self, sample_qz, sample_pz):
        opts = self.opts
        sigma2_p = opts['pz_scale'] ** 2
        n = utils.get_batch_size(sample_qz)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods


        # k(x, y) = C / (C + ||x - y||^2)
        # C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        # C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
        Cbase = 2. * opts['zdim'] * sigma2_p

        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
        return stat


    @staticmethod
    def reconstruction_loss(opts, real, reconstr):
        # real = self.sample_points
        # reconstr = self.reconstructed
        if opts['cost'] == 'l2':
            # c(x,y) = ||x - y||_2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
            loss = 0.2 * tf.reduce_mean(tf.sqrt(1e-08 + loss))
        elif opts['cost'] == 'l2sq':
            # c(x,y) = ||x - y||_2^2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
            loss = 0.05 * tf.reduce_mean(loss)
        elif opts['cost'] == 'l1':
            # c(x,y) = ||x - y||_1
            loss = tf.reduce_sum(tf.abs(real - reconstr), axis=[1, 2, 3])
            loss = 0.02 * tf.reduce_mean(loss)
        else:
            assert False, 'Unknown cost function %s' % opts['cost']
        return loss



    def optimizer(self, lr, decay=1.):
        opts = self.opts
        lr *= decay
        # with tf.variable_scope("main/t_network/t_network_3/t_3_1_0", reuse=True):
        #     w1 = tf.get_variable("W", [8, 1000])
        # self.grad = tf.gradients(self.trans_loss, w1)
        # print('=======================================')
        # print(self.grad)
        if opts["optimizer"] == "sgd":
            return tf.train.GradientDescentOptimizer(lr)
        elif opts["optimizer"] == "adam":
            return tf.train.AdamOptimizer(lr, beta1=opts["adam_beta1"])
        else:
            assert False, 'Unknown optimizer.'

    def add_optimizers(self):
        opts = self.opts
        lr = opts['lr']
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        ae_vars = encoder_vars + decoder_vars
        
        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        t_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main/t_network')
        self.t_network_vars = t_network_vars


        if opts['verbose']:
            logging.error('Param num in AE: %d' % \
                    np.sum([np.prod([int(d) for d in v.get_shape()]) \
                    for v in ae_vars]))

        # Auto-encoder optimizer
        clip_grad = 3
        opt = self.optimizer(lr, self.lr_decay)
        ae_grads, ae_variables = zip(*opt.compute_gradients(self.wae_loss, encoder_vars + decoder_vars))
        ae_grads, ae_global_norm = tf.clip_by_global_norm(ae_grads, clip_grad)
        self.ae_opt = opt.apply_gradients(zip(ae_grads, ae_variables))
        # self.ae_opt = opt.minimize(loss=self.wae_loss,
        #                       var_list=encoder_vars + decoder_vars)

        # gradients___ = tf.gradients(self.trans_loss, main_vars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        
        # pdb.set_trace()
        # for grad, var in zip(gradients___, main_vars):
        #     if grad == None:
        #         print('[FUNK]:{}'.format(var.name))
        #     else:
        #         print("Great: {}".format(var.name))
        # assert 0

        # self.first_main_opt = opt.minimize(loss=self.main_loss, var_list = main_vars)
        main_grads, main_variables = zip(*opt.compute_gradients(self.main_loss, main_vars))
        main_grads, main_global_norm = tf.clip_by_global_norm(main_grads, clip_grad)
        self.main_opt = opt.apply_gradients(zip(main_grads, main_variables))
        # self.main_opt = opt.minimize(loss=self.main_loss, var_list = main_vars)

        # Discriminator optimizer for WAE-GAN
        self.z_adv_opt = None

        # Encoder optimizer
        if opts['e_pretrain']:
            opt = self.optimizer(lr)
            self.pretrain_opt = opt.minimize(loss=self.loss_pretrain,
                                             var_list=encoder_vars)
        else:
            self.pretrain_opt = None

    def sample_pz(self, num=100, seed=None):
        opts = self.opts
        noise = None
        distr = opts['pz']
        if not seed == None:
            np.random.seed(seed)
        if distr == 'uniform':
            noise = np.random.uniform(
                -1, 1, [num, opts["zdim"]]).astype(np.float32)
        elif distr in ('normal', 'sphere'):
            mean = np.zeros(opts["zdim"])
            cov = np.identity(opts["zdim"])
            noise = np.random.multivariate_normal(
                mean, cov, num).astype(np.float32)
            if distr == 'sphere':
                noise = noise / np.sqrt(
                    np.sum(noise * noise, axis=1))[:, np.newaxis]
        return opts['pz_scale'] * noise

    def pretrain_encoder(self, data):
        opts = self.opts
        steps_max = 200
        batch_size = opts['e_pretrain_sample_size']
        for step in range(steps_max):
            train_size = data.num_points
            data_ids = np.random.choice(train_size, min(train_size, batch_size),
                                        replace=False)
            batch_images = data.data[data_ids].astype(np.float)
            batch_noise =  self.sample_pz(num=batch_size)

            [_, loss_pretrain] = self.sess.run(
                [self.pretrain_opt,
                 self.loss_pretrain],
                feed_dict={self.sample_points: batch_images,
                           self.sample_noise: batch_noise,
                           self.is_training: True})

            if opts['verbose']:
                logging.error('Step %d/%d, loss=%f' % (
                    step, steps_max, loss_pretrain))

            if loss_pretrain < 0.1:
                break



    def train(self, dataset, task_num, random_z_all, pseudo_G_z_all,
              pseudo_T_z_all, w_old, b_old, f_g_z_all, pred_all, lambda_list):
        
        opts = self.opts
        
        losses = []
        losses_rec = []
        losses_entropy = []
        losses_match = []
        losses_trans = []
        losses_reg = []
        losses_f = []

        blurr_vals = []
        encoding_changes = []
        enc_test_prev = None
        
        batch_size = opts['batch_size']
        train_size = dataset[task_num].num_points
        batches_num = int(train_size / batch_size)
        
        self.num_pics = opts['plot_num_pics']
        self.fixed_noise = self.sample_pz(num=opts['plot_num_pics'])


        if task_num == 0:
            logging.error('Pretraining the encoder')
            self.pretrain_encoder(dataset[task_num])
            logging.error('Pretraining the encoder done.')

        self.start_time = time.time()
        
        decay = 1.
        wae_lambda = lambda_list['wae_lambda']
        rec_lambda = lambda_list['rec_lambda']
        trans_lambda = lambda_list['trans_lambda']
        reg_lambda = lambda_list['reg_lambda']
        f_lambda = lambda_list['f_lambda']
        main_lambda = lambda_list['main_lambda']

        main_keep_prob = opts['main_keep_prob']
        t_keep_prob = opts['t_keep_prob']

        '''
        real_blurr = self.sess.run(
            self.blurriness,
            feed_dict={self.sample_points: data.data[:self.num_pics]})
        logging.error('Real pictures sharpness = %.5f' % np.min(real_blurr))
        '''
        
        for epoch in range(opts["epoch_num"][task_num]):
            counter = 0
            acc = 0.0
            
            # Update learning rate if necessary

            if opts['lr_schedule'] == "manual":
                if epoch == 30:
                    decay = decay / 2.
                if epoch == 50:
                    decay = decay / 5.
                if epoch == 100:
                    decay = decay / 10.

            # Iterate over batches

            for it in range(batches_num):

                # Sample batches of data points and Pz noise
                # batch data in sequence
                '''
                ist = it * batch_size
                ied = min((it + 1) * batch_size, train_size)

                batch_images = data.data[ist:ied].astype(np.float)
                batch_labels = data.labels[ist:ied]
                batch_noise = self.sample_pz(opts['batch_size'])
                '''
                
                
                # batch data randomly chosen
                # seed = opts['seed'][1] + it
                # np.random.seed(seed)
                data_ids = np.random.choice(
                    train_size, opts['batch_size'], replace=False)
                batch_images = dataset[task_num].data[data_ids].astype(np.float)
                batch_labels = dataset[task_num].labels[data_ids]

                # batch_noise = self.sample_pz(num=opts['batch_size'], seed=opts['seed'][3] + it)
                batch_noise = self.sample_pz(num=opts['batch_size'])
                
                
                # Randomly choose 'sample_size' amount of 'random_z'
                # And calculate corresponding parameters
                # seed = opts['seed'][2] + it
                # np.random.seed(seed)
                idx = np.random.choice(opts['z_size'], opts['sample_size'], replace=False)
                pred = pred_all[idx]
                random_z = random_z_all[idx]
                f_g_z = f_g_z_all[idx]
                pseudo_G_z = pseudo_G_z_all[idx]
                pseudo_T_z = []
                for l in range(len(opts['auxi_info'])):
                    if opts['auxi_info'][l]:
                        pseudo_T_z.append(pseudo_T_z_all[l][idx])
                    else:
                        pseudo_T_z.append(None)

                # Update encoder and decoder
                feed_d = {self.sample_noise: batch_noise,
                        self.sample_points: batch_images,
                        self.sample_labels: batch_labels,
                        
                        self.random_z: random_z,
                        self.pseudo_G_z: pseudo_G_z,
                        self.f_g_z: f_g_z,

                        
                        self.lr_decay: decay,
                        self.wae_lambda: wae_lambda,
                        self.rec_lambda: rec_lambda,
                        self.trans_lambda: trans_lambda,
                        self.reg_lambda: reg_lambda,
                        self.f_lambda: f_lambda,
                        self.main_lambda: main_lambda,

                        self.main_keep_prob: main_keep_prob,
                        self.t_keep_prob: t_keep_prob,
                        self.is_training: True,
                        self.false: False}
                
                for l in range(len(opts['auxi_info'])):
                    if opts['main_info'][l] - opts['auxi_info'][l]:
                        feed_d[self.w_old[l]] = w_old[l]
                    feed_d[self.b_old[l]] = b_old[l]
                    if  opts['auxi_info'][l]:
                        feed_d[self.pseudo_T_z[l]] = pseudo_T_z[l]
                        # new_auxi_weights_l = self.sess.run(self.auxi_weights_list[l],
                        #                         feed_dict={self.encoded: random_z,
                        #                                     self.t_keep_prob: 1.0})
                        # feed_d[self.new_auxi_weights_list[l]] = new_auxi_weights_l

                self.sess.run(self.ae_opt, feed_dict=feed_d)
                self.sess.run(self.ae_opt, feed_dict=feed_d)
                
                # if task_num == 0:
                #     self.sess.run(self.ae_opt, feed_dict=feed_d)
                #     real_opt = self.first_main_opt
                # else:
                #     real_opt = self.main_opt
                    
                [ _, loss, loss_rec, loss_entropy, loss_match,
                 loss_trans, loss_reg, loss_f, accuracy] = self.sess.run( # 
                    [self.main_opt,
                     self.main_loss,
                     self.loss_reconstruct,
                     self.cross_entropy,
                     self.penalty,
                     self.trans_loss,
                     self.reg_loss,
                     self.f_loss,
                     self.accuracy],
                    feed_dict=feed_d)

                
                #logging.error('loss_reg :%.5f' % loss_reg)
                losses.append(loss)
                losses_rec.append(loss_rec)
                losses_entropy.append(loss_entropy)
                losses_match.append(loss_match)
                losses_trans.append(loss_trans)
                losses_reg.append(loss_reg)
                losses_f.append(loss_f)
                #if opts['verbose']:
                #    logging.error('Matching penalty after %d steps: %f' % (
                #        counter, losses_match[-1]))

                acc += accuracy
                #logging.error('Accuracy after %d steps: %f' % (counter, accuracy))

                counter += 1

                # Print debug info

                # if counter % opts['print_every'] == 0:
                if it == batches_num - 1:
                    now = time.time()

                    # Printing various loss values

                    debug_str = 'Epoch: %d/%d, batch:%d/%d, ' % (
                        epoch + 1, opts['epoch_num'][task_num],
                        it + 1, batches_num)
                    debug_str += 'accuracy: %.5f \n' % accuracy

                    # 'reg_loss=%.5f, ' \  'f_loss=%.5f'
                    debug_str += 'loss=%.5f, recon_loss=%.5f, ' \
                                 'cross_entropy=%.5f, ' \
                                 'match_loss=%.5f, ' \
                                 'trans_loss=%.5f, ' \
                                 'reg_loss=%.5f, ' \
                                 'f_loss=%.5f' % (
                                    losses[-1], rec_lambda * losses_rec[-1], main_lambda * losses_entropy[-1],
                                    wae_lambda * losses_match[-1], trans_lambda * losses_trans[-1],
                                    reg_lambda * losses_reg[-1], f_lambda * losses_f[-1]) # 
                    logging.error(debug_str)

                    self.test(dataset)


        
    def test(self, dataset, is_last=False):
        opts = self.opts
        batch_size = opts['batch_size']
        accs = []
        for i in range(len(dataset)):
            batches_num = int(dataset[i].test_num_points / batch_size)
            data_size = dataset[i].test_num_points
            acc = 0.0
            for step in range(batches_num):
                ist = step * batch_size
                ied = (step + 1) * batch_size
                accuracy = self.sess.run(self.accuracy, 
                                    feed_dict={ self.sample_points: dataset[i].test_data[ist:ied], 
                                                self.sample_labels: dataset[i].test_labels[ist:ied],
                                                self.main_keep_prob: 1.0,
                                                self.t_keep_prob: 1.0,
                                                self.is_training: False})
                acc += (ied - ist) * accuracy

                # if i == 0 and step == 0:
                #     new_auxi_weights = self.sess.run(self.auxi_weights_list[-1],
                #                                 feed_dict={self.sample_points: dataset[i].test_data[ist:ied],
                #                                             self.t_keep_prob: 1.0,
                #                                             self.is_training: False})
                #     logging.error("T z")
                #     logging.error(new_auxi_weights)
                #     with tf.variable_scope("main/main_1", reuse=True):
                #         w1 = tf.get_variable("W", [784, 600])
                #     logging.error(self.sess.run(tf.gradients(self.trans_loss, w1)), 
                #                                 feed_dict={self.sample_points: dataset[i].test_data[ist:ied],
                #                                             self.sample_labels: dataset[i].test_labels[ist:ied],
                #                                             self.main_keep_prob: 1.0,
                #                                             self.t_keep_prob: 1.0,
                #                                             self.is_training: False})


            acc /= data_size
            accs.append(acc)
            if is_last == False:
                acc_str = "TASK %d ACCURACY: %.7f" % (i, acc)
                logging.error(acc_str)
            else:
                acc_str = "Taks %d acc: %.7f" % (i, acc)
                logging.error(acc_str)
        if is_last == False:
            logging.error("Average accuracy: %.7f" % (sum(accs) / len(accs)))
        else:
            logging.error("Average acc: %.7f" % (sum(accs) / len(accs)))




    
    def former(self):

        opts = self.opts
        # Randomly sample an "z_size" amount of z, for example 10000.
        seed = opts["seed"][0]
        # random_z = self.sample_pz(num=opts['z_size'], seed=seed)
        random_z = self.sample_pz(num=opts['z_size'])
        # Auxiliary weights T(z) generated from "random_z"
        pseudo_T_z = []
        # Weights and biases of the main network
        w_old = []
        b_old = []
        
        for b in range(10):
            bn = int(opts['z_size'] / 10)
            st = b * bn
            ed = (b + 1) * bn
            # Images generated from "random_z"
            b_pseudo_G_z = self.sess.run(self.decoded,
                                   feed_dict={self.sample_noise: random_z[st:ed],
                                              self.is_training: False})
            b_f_g_z = self.sess.run(self.encoded,
                               feed_dict={self.sample_points: b_pseudo_G_z,
                                          self.is_training: False})
            b_pred = self.sess.run(self.prediction,
                                feed_dict={self.sample_points: b_pseudo_G_z,
                                            self.is_training: False,
                                            self.t_keep_prob: 1.0,
                                            self.main_keep_prob: 1.0})

            if b == 0:
                pseudo_G_z = b_pseudo_G_z
                f_g_z = b_f_g_z
                pred = b_pred
            else:
                pseudo_G_z = np.concatenate((pseudo_G_z, b_pseudo_G_z))
                f_g_z = np.concatenate((f_g_z, b_f_g_z))
                pred = np.concatenate((pred, b_pred))
            
            for l in range(len(opts['auxi_info'])):
                if opts['auxi_info'][l]:
                    pseudo_auxi_weights = self.sess.run(self.auxi_weights_list[l],
                                                    feed_dict={self.encoded: random_z[st:ed],
                                                                self.t_keep_prob: 1.0})
                    if b == 0:
                        pseudo_T_z.append(pseudo_auxi_weights)
                    else:
                        pseudo_T_z[l] = np.concatenate((pseudo_T_z[l], pseudo_auxi_weights))
                else:
                    pseudo_auxi_weights = None
                    if b == 0:
                        pseudo_T_z.append(pseudo_auxi_weights)
        
        for l in range(len(opts['auxi_info'])):    
            with tf.variable_scope('main/main_%d' % (l+1), reuse=True):
                if opts['main_info'][l] - opts['auxi_info'][l]:
                    weight = self.sess.run(tf.get_variable('W'))
                    w_old.append(weight)
                else:
                    w_old.append(None)

                bias = self.sess.run(tf.get_variable('b'))
                b_old.append(bias)
            
        
        return random_z, pseudo_G_z, pseudo_T_z, w_old, b_old, f_g_z, pred
