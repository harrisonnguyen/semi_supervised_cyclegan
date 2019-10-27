from model.cyclegan import CycleGAN
from model.network import d_layer, generator
from tensorflow.keras.layers import Flatten,Dense, Input
from tensorflow.keras.models import Model
from keras import backend as K
import tensorflow as tf
from model_utils import learning_utils as learning
import numpy as np
class SemiWassersteinCycleGAN(CycleGAN):
    def __init__(self,mse_weight=1.0,gradient_penalty_scale=3.0,*args,**kwargs):
        self._ALPHA = mse_weight
        self._SCALE = gradient_penalty_scale
        super(SemiWassersteinCycleGAN,self).__init__(*args,**kwargs)

    def _build_graph(self,*args,**kwargs):
        gf = args[0]
        df = args[1]
        depth = args[2]
        patch_size = args[3]
        n_modality = args[4]
        self._xphA = tf.placeholder(tf.float32,
                                    [None,patch_size,patch_size,n_modality])
        self._xphB = tf.placeholder(tf.float32,
                                    [None,patch_size,patch_size,n_modality])


        self._batch_step = tf.Variable(0,trainable=False,dtype=tf.int32)
        self._batch_step_inc = tf.assign_add(self._batch_step,1)
        self._epoch = tf.Variable(0,trainable=False,dtype=tf.int32)
        self._epoch_inc = tf.assign_add(self._epoch,1)

        self.g_AB = generator(self.img_shape,gf,depth)
        self.g_BA = generator(self.img_shape,gf,depth)

        # translate images to new domain
        self._predictedB = self.g_AB(self._xphA)
        self._predictedA = self.g_BA(self._xphB)

        # translate to original domain
        self._reconstructA = self.g_BA(self._predictedB)
        self._reconstructB = self.g_AB(self._predictedA)

        # identity mappigs
        self._img_A_id = self.g_BA(self._xphA)
        self._img_B_id = self.g_AB(self._xphB)

        self.d_A = critic(self.img_shape,df,depth)
        self.d_B = critic(self.img_shape,df,depth)

        self._realA = self.d_A(self._xphA)
        self._fakeA = self.d_A(self._predictedA)
        self._realB = self.d_B(self._xphB)
        self._fakeB = self.d_B(self._predictedB)

        # wasserstein
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hatA = epsilon * self._xphA + (1 - epsilon) * self._predictedA
        d_hatA = self.d_A(x_hatA)
        self._ddxA = tf.gradients(d_hatA, x_hatA)[0]

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hatB = epsilon * self._xphB + (1 - epsilon) * self._predictedB
        d_hatB = self.d_B(x_hatB)
        self._ddxB = tf.gradients(d_hatB, x_hatB)[0]

    def _create_loss(self,*args,**kwargs):
        super(SemiWassersteinCycleGAN,self)._create_loss(*args,**kwargs)
        self._reconstructA_loss = tf.losses.absolute_difference(self._xphA,self._reconstructA)

        self._reconstructB_loss = tf.losses.absolute_difference(self._xphB,self._reconstructB)

        self._cycle_loss = self._reconstructA_loss + self._reconstructB_loss

        # improved wasserstein loss
        self._discrimB_loss = tf.reduce_mean(self._realB) - tf.reduce_mean(self._fakeB)
        self._discrimA_loss = tf.reduce_mean(self._realA) - tf.reduce_mean(self._fakeA)

        self._genA_loss = tf.reduce_mean(self._fakeA) + self._cycle_loss
        self._genB_loss = tf.reduce_mean(self._fakeB) + self._cycle_loss


        # wasserstein gradient penalty
        self._ddxA = tf.sqrt(tf.reduce_sum(tf.square(self._ddxA), axis=1))
        self._ddxA = tf.reduce_mean(tf.square(self._ddxA - 1.0) * self._SCALE)

        self._ddxB = tf.sqrt(tf.reduce_sum(tf.square(self._ddxB), axis=1))
        self._ddxB = tf.reduce_mean(tf.square(self._ddxB - 1.0) * self._SCALE)

        self._discrimA_loss += self._ddxA
        self._discrimB_loss += self._ddxB

        shape = self._xphA.get_shape().as_list()
        self._pair_loss_ph = tf.placeholder(tf.int32,
            [None,shape[1],shape[2],shape[3]])
        self._mseA = tf.losses.mean_squared_error(predictions=self._predictedA,
                                    labels=self._xphA,
                                    weights=self._pair_loss_ph)
        self._mseB = tf.losses.mean_squared_error(predictions=self._predictedB,
                                    labels=self._xphB,
                                    weights=self._pair_loss_ph)

        self._genA_loss += self._ALPHA*self._mseA

        self._genB_loss += self._ALPHA*self._mseB

        self._reconstruction_loss = tf.losses.mean_squared_error(predictions=self._predictedB,
                                                labels=self._xphB)
    def _create_summary(self,*args,**kwargs):
        super(SemiWassersteinCycleGAN,self)._create_summary(*args,**kwargs)

    def _create_optimiser(self,*args,**kwargs):
        super(SemiWassersteinCycleGAN,self)._create_optimiser(*args,**kwargs)
        variables = [self._mseA,
                    self._mseB,
                    self._xphA,
                    self._xphB]
        types = ['scalar']*2+['image']*2
        names = ['loss/mseA','loss/mseB','image/x_phA_paired','image/x_phB_paired']

        self._summary_pair_op,self._weights_pair_op = learning.create_summary(variables,types,names)

    def train_step(self,A,B,A_paired,B_paired,
                    write_summary=False):
        mask = np.zeros_like(A)
        data={self._xphA: A,
              self._xphB: B,
              self._pair_loss_ph: mask,
              K.learning_phase():True}
        summary,global_step,_,_ =self.sess.run([self._summary_op,
                                            self._batch_step,
                                            self._solver,
                                            self._batch_step_inc],
                                            feed_dict=data)
        mask = np.ones_like(A_paired)
        data={self._xphA: A_paired,
              self._xphB: B_paired,
              self._pair_loss_ph: mask,
              K.learning_phase():True}
        summary_pair,_ =self.sess.run([self._summary_pair_op,
                                    self._solver],
                                    feed_dict=data)
        if write_summary:
            self._train_writer.add_summary(summary, global_step)
            self._train_writer.add_summary(summary_pair, global_step)
        return global_step

    def __str__(self):
        return "mse"

def critic(input_img_shape,df,depth):
    img = Input(shape=input_img_shape)

    d1 = d_layer(img, df, normalization=False)
    input = d1
    for i in range(1,depth):
        output = d_layer(input, df*2**i)
        input = output
    output = Flatten()(input)
    validity = Dense(1)(output)

    return Model(img, validity)
