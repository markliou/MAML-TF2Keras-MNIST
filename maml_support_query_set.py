import numpy as np 
import tensorflow as tf # use the tf2.4
import tensorflow_datasets as tfds 
import tensorflow_addons as tfa
from adabelief_tf import AdaBeliefOptimizer

inner_lr = 1E-3
outter_lr = 1E-3
inner_task_loop_no = 32 # give 32 tasks of 2ways-1shot
# opt_inner = tf.keras.optimizers.Adagrad(inner_lr)
# opt_outter = tf.keras.optimizers.Adagrad(outter_lr, clipnorm=1.)
# opt_inner = tfa.optimizers.NovoGrad(inner_lr)
# opt_outter = tfa.optimizers.NovoGrad(outter_lr, clipnorm=1.)
# opt_inner = tfa.optimizers.RectifiedAdam(inner_lr)
# opt_outter = tfa.optimizers.RectifiedAdam(outter_lr, clipnorm=1.)
opt_inner = AdaBeliefOptimizer(inner_lr)
opt_outter = AdaBeliefOptimizer(outter_lr)

def cnn():
    Input = tf.keras.Input([28, 28, 1])
    Input_n = Input/128.0 - 1
    conv1 = tf.keras.layers.Conv2D(32, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu)(Input_n) #[14,14]
    conv2 = tf.keras.layers.Conv2D(32, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu)(conv1) #[7,7]
    conv3 = tf.keras.layers.Conv2D(32, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu)(conv2) #[4,4]
    fc = tf.keras.layers.Flatten()(conv3)
    fc1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(fc)
    fc2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(fc1)
    fc3 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(fc2)
    out = tf.keras.layers.Dense(1, activation=None)(fc3)
    return tf.keras.Model(inputs=Input, outputs=out)
pass 

def select_support_query_set(dataset_iter):
    flag_1 = flag_0 = True
    imgs_1_support, labs_1_support = next(dataset_iter)
    imgs_0_support, labs_0_support = next(dataset_iter)
    support_set = tf.concat([imgs_1_support, imgs_0_support], axis=0)
    while(flag_1 or flag_0):
        imgs_query, labs_query = next(dataset_iter)
        if ((labs_query.numpy() == labs_1_support.numpy()) and flag_1) :
            imgs_1_query = tf.Variable(imgs_query)
            flag_1 = False
        elif ((labs_query.numpy() == labs_0_support.numpy()) and flag_0) :
            imgs_0_query = tf.Variable(imgs_query)
            flag_0 = False
        pass 
    pass 
    query_set = tf.concat([imgs_1_query, imgs_0_query], axis=0)
    return [support_set, query_set]
pass 

(kmnist_tr, kmnist_ts) = tfds.load('kmnist', 
                                   split=['train','test'],
                                   shuffle_files=True,
                                   as_supervised=True,
                                   with_info=False,)

# Task hint:
# this example will use 2ways-1shot for training the MAML.
# the inner task loop will set as 30. 
kmnist_tr = kmnist_tr.batch(1).repeat()
kmnist_tr_iter = iter(kmnist_tr)
meta_labs = tf.Variable([[1], [0]], dtype=tf.float32)

cnn_model = cnn()

# according to the TF2 manual, the variable of the model would need to be
# initialized, and use it to inferece something is a way to initialize 
# the varialbe.
support_set, query_set = select_support_query_set(kmnist_tr_iter)
_ = cnn_model(support_set) # 1)for initializing the model. 2)for define the loss function out-of the loop 
loss_inner_meta = lambda: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=meta_labs, logits=cnn_model(support_set)))
loss_inner_task = lambda: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=meta_labs, logits=cnn_model(query_set)))

for step in range(5000):
    # keep the original weights for meta-training
    meta_weights = [tf.Variable(target_weights) for target_weights in cnn_model.trainable_weights]
    
    def meta_loss():
        loss_outter = 0
        for task_count in range(inner_task_loop_no):
            # fetch the dataset. The first dataset is always be 1. 
            # So, the labels will be defined as the meta-labels insteadly. 
            support_set, query_set = select_support_query_set(kmnist_tr_iter)
            
            # put the meta-weights into the model
            for model_weight_index in range(len(cnn_model.trainable_weights)):
                cnn_model.trainable_weights[model_weight_index].assign(meta_weights[model_weight_index])
            pass

            # updateing the weights from meta-weights
            opt_inner.minimize(loss=loss_inner_meta, var_list=cnn_model.trainable_weights) 
            
            # calculating the task loss (after the meta-weights updating)
            # here, the same support set and query set are used. you can 
            # also creat a different query set for meta-weight training.
            loss_outter += loss_inner_task()
        pass 
        return loss_outter/inner_task_loop_no
    pass 

    # put the meta-weights into the model
    for model_weight_index in range(len(cnn_model.trainable_weights)):
        cnn_model.trainable_weights[model_weight_index].assign(meta_weights[model_weight_index])
    pass

    # update the meta-weights
    opt_outter.minimize(loss=meta_loss, var_list=cnn_model.trainable_weights) 

    print("step: {}  outter_loss: {}".format(step, meta_loss()))
pass 
