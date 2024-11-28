#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adapted from https://github.com/renatolfc/chimera-stf

from __future__ import print_function
import time
import logging
import scipy.io
from collections import deque
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import datetime
import pickle


def stf(A_t, C_t, K=10, steps=1000, learning_rate=0.001, beta=0.001, lambda1=0.005,
        c0_scaler=0.01, c1_scaler=1.0, unobserved_C=0,delta=1.,name='',
        lambda2=0.001, logdir='logs/', early_stopping=10, es_tolerance=100000):

    tf.reset_default_graph()
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    modelname = f'{name}_{dt}'

    logging.info('Running Shared Matrix Factorization over time')

    if A_t.shape[0] != C_t.shape[0]:
        raise ValueError(
            'stf was called with matrices with different dimensions in the '
            'time axis.'
        )

    T = A_t.shape[0]
    N = A_t.shape[1]
    M = A_t.shape[-1]
    D = C_t.shape[-1]

    logging.info(
        f'name= {modelname}, k = {K}, steps = {steps}, learning_rate = {learning_rate}, beta = {beta}, lambda1 = {lambda1}, lambda2 = {lambda2}'
    )

    tUs = tf.Variable(
        tf.random.normal(
            [T, N, K],
            mean=0.0,stddev=1,dtype=tf.float32),
         name='U', dtype=tf.float32
    )

    tVs = tf.Variable(
        tf.random.normal(
            [1,M,K],
            mean=0.0, stddev=1, dtype=tf.float32),
         name='V', dtype=tf.float32
    )

    tWs = tf.Variable(
        tf.random.normal(
            [1,D,K],
            mean=0.0,stddev=1,dtype=tf.float32),
         name='W', dtype=tf.float32
    )

    V_bias = tf.Variable(
        tf.random.normal(
            [T, M, 1],
            mean=0.0,stddev=1,dtype=tf.float32),
        name='V_bias', dtype=tf.float32
    )

    U_bias = tf.Variable(
        tf.random.normal(
            [T, N, 1],
            mean=0.0,stddev=1,dtype=tf.float32),
         name='U_bias', dtype=tf.float32
    )

    W_bias = tf.Variable(
        tf.random.normal(
            [T, D, 1],
            mean=0.0,stddev=1,dtype=tf.float32),
        name='W_bias', dtype=tf.float32
    )

    pAt = tf.placeholder(
        tf.float32,
        [T, N, M],
        name='A'
    )

    pCt = tf.placeholder(
        tf.float32,
        [T, N, D],
        name='C'
    )

    lr = tf.placeholder(tf.float32, name='learning_rate')
    c0 = tf.placeholder(tf.float32, name='scaling_factor_0')
    c1 = tf.placeholder(tf.float32, name='scaling_factor_1')
    c0_val = tf.placeholder(tf.float32, name='unobserved_C_val')

    tUs_tVs = tf.matmul(tUs, tf.transpose(tVs, [0, 2, 1])) \
              + U_bias + \
              tf.transpose(V_bias, [0, 2, 1])
    tUs_tWs = tf.matmul(tUs, tf.transpose(tWs, [0, 2, 1])) \
              + U_bias + \
              tf.transpose(W_bias, [0, 2, 1])

    pAt_tUs_tVs = pAt - tUs_tVs
    pCt_tUs_tWs = pCt - tUs_tWs

    logging.info(
        f'name= {modelname}, k = {K}, steps = {steps}, learning_rate = {learning_rate}, '
        f'beta = {beta}, lambda1 = {lambda1}, lambda2 = {lambda2}'
    )

    # Losses
    # Mask for absent users
    empty_users = 1. - tf.cast(tf.equal(tf.reduce_sum(pAt, axis=2), 0),tf.float32)  # 1 if any interactions, 0 otherwise

    # A losses
    ## scale losses from empty As
    obs_mask_A = tf.cast(tf.math.equal(pAt, 0.), tf.float32) # 1 if ==0, 0 otherwise
    obs_scaling_A = obs_mask_A * c0 + (1. - obs_mask_A) * c1
    loss_mask_A = tf.multiply(tf.expand_dims(empty_users, 2), obs_scaling_A)

    ## total loss from A
    tA_tmp = tf.square(pAt_tUs_tVs)
    tA_loss = delta*tf.reduce_sum(tf.multiply(tA_tmp, loss_mask_A))

    # C losses
    tC_tmp = tf.square(pCt_tUs_tWs)
    ## scale losses from empty Cs
    obs_mask_C = tf.cast(tf.math.equal(pCt, c0_val), tf.float32)
    obs_scaling_C = obs_mask_C * c0 + (1. - obs_mask_C) * c1
    loss_mask_C = tf.multiply(tf.expand_dims(empty_users, 2), obs_scaling_C)
    tC_loss = beta * tf.reduce_sum(tf.multiply(tC_tmp, loss_mask_C))     ## total loss from C

    #regularisation losses
    # weight size regularisation
    reg1 = lambda1 * (
        tf.reduce_sum(tf.square(tWs)) +
        tf.reduce_sum(tf.square(tVs)) +
        tf.reduce_sum(tf.multiply(tf.expand_dims(empty_users, 2), tf.square(tUs)))
                      )

    ## timestep similarity regularization
    reg2 = lambda2 * (tf.reduce_sum(tf.square(tUs[1:] - tUs[:T - 1]))
                  + tf.reduce_sum(tf.square(tWs[1:] - tWs[:T - 1]))
                  + tf.reduce_sum(tf.square(tVs[1:] - tVs[:T - 1])))

    loss = tA_loss + tC_loss + reg2 + reg1

    # Losses }}}

    optimizer = tf.train.AdamOptimizer(lr).minimize(loss, var_list=[tUs, tVs, tWs, U_bias, V_bias, W_bias])

    if logdir:
        logdir = logdir + modelname
        writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
    else:
        writer = None

    tf.summary.scalar("loss", loss)
    tf.summary.scalar('A_loss', tA_loss)
    tf.summary.scalar("C_loss", tC_loss)
    tf.summary.scalar("reg1", reg1)
    tf.summary.scalar("reg2", reg2)

    error = 1000
    summary_op = tf.summary.merge_all()
    loss_history = deque(maxlen=early_stopping + 1)

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())

        start_time = time.time()
        for step in range(steps):
            error, summary, _, = sess.run(
                [loss, summary_op, optimizer],
                feed_dict={
                    pAt: A_t,
                    pCt: C_t,
                    lr: learning_rate,
                    c0: c0_scaler,
                    c1: c1_scaler,
                    c0_val: unobserved_C
                }
            )

            if writer:
                writer.add_summary(summary, step)

            logging.info('Current iteration: %d, current loss: %g',
                          step, error)
            loss_history.append(error)

            if len(loss_history) > early_stopping:
                if loss_history.popleft() - min(loss_history) < es_tolerance:
                    logging.info(f'\nEarly stopping. No validation loss improvement in {early_stopping} epochs.')
                    break

            if step in [100, 1000, 3000, 5000]:
                Us = sess.run(tUs)
                Vs = sess.run(tVs)
                Ws = sess.run(tWs)
                Ub = sess.run(U_bias)
                Vb = sess.run(V_bias)
                Wb = sess.run(W_bias)

                res = {'loss': error, 'Us': Us, 'Vs': Vs,
                       'Ws': Ws, 'U_bias': Ub, 'V_bias': Vb, 'W_bias': Wb}
                res['config'] = {'lr': learning_rate, 'c0': c0_scaler, 'c1': c1_scaler,
                                 'c0_val': unobserved_C, 'lambda_1': lambda1, 'lambda_2':
                                  lambda2, 'beta': beta, 'delta':delta}
                with open(f'../models/{modelname}_{step}.pkl', 'wb') as f:
                    pickle.dump(res, f)
                print(f'Saving intermediate model to file: {modelname}_{step}.pkl')

        end_time = time.time()

        logging.info('Optimization ran for %g s and final loss is %g',
                     end_time - start_time, error)

        Us = sess.run(tUs)
        Vs = sess.run(tVs)
        Ws = sess.run(tWs)
        Ub = sess.run(U_bias)
        Vb = sess.run(V_bias)
        Wb = sess.run(W_bias)

    res = {'loss': error,'Us': Us,'Vs': Vs,
       'Ws': Ws,'U_bias': Ub,'V_bias': Vb,'W_bias': Wb}

    res['config'] = {'lr': learning_rate,'c0': c0_scaler,'c1': c1_scaler,
        'c0_val': unobserved_C,'lambda_1': lambda1,'lambda_2': lambda2,'beta': beta, 'delta':delta}

    with open(f'../models/{modelname}_full.pkl', 'wb') as f:
        pickle.dump(res, f)
    print(f'Saving to file: models/{modelname}_full.pkl')

    return res
