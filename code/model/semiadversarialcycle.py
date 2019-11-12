import tensorflow as tf
from model.cyclegan import CycleGAN
from model.network import discriminator
from model_utils import learning_utils as learning
from keras import backend as K
from model.network import d_layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D

def pair_discriminator(input_img_shape):
    img = Input(shape=input_img_shape)

    output = d_layer(img, int(input_img_shape[-1]/2))
    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(output)

    return Model(img, validity,name="d_pair")

class SemiAdverCycleGAN(CycleGAN):
    def __init__(self,entropy_weight=1.0,*args,**kwargs):
        self._ALPHA = entropy_weight
        super(SemiAdverCycleGAN,self).__init__(*args,**kwargs)


    def _build_graph(self,*args,**kwargs):
        super(SemiAdverCycleGAN,self)._build_graph(*args,**kwargs)
        print(self.d_A.summary())
        patch_size = args[3]
        n_modality = args[4]
        df = args[1]
        depth= args[2]
        self._xphA_paired = tf.placeholder(tf.float32, [None,patch_size,patch_size,n_modality])
        self._xphB_paired = tf.placeholder(tf.float32, [None,patch_size,patch_size,n_modality])

        # we concatenate the paired images on the channel axis
        img_shape = self.img_shape
        #img_shape[-1] = self.img_shape[-1]*2
        img_shape[-1] = int(df*2**(depth))
        img_shape[-2] = int(self.img_shape[-2]/2**depth)
        img_shape[-3] = int(self.img_shape[-3]/2**depth)

        self._predictedB_pair = self.g_AB(self._xphA_paired)
        self._predictedA_pair = self.g_BA(self._xphB_paired)

        # share weights with other discrim
        intermediate_layerA = Model(inputs=self.d_A.input,
                                 outputs=self.d_A.layers[-2].output)
        intermediate_layerB = Model(inputs=self.d_B.input,
                                 outputs=self.d_B.layers[-2].output)
        fmap_apair = intermediate_layerA(self._xphA_paired)
        fmap_afake = intermediate_layerA(self._predictedA_pair)

        fmap_bpair = intermediate_layerB(self._xphB_paired)
        fmap_bfake = intermediate_layerB(self._predictedB_pair)
        self.d_pair = pair_discriminator(img_shape)
        print(self.d_pair.summary())
        #self.d_pair = discriminator(img_shape,df,depth)


        """
        paired_real = tf.concat([self._xphA_paired),
                                self._xphB_paired],
                                axis=-1)
        paired_fakeA = tf.concat([self._predictedA,
                                self._xphB_paired],
                                axis=-1)
        paired_fakeB = tf.concat([self._xphA_paired,
                                self._predictedB],
                                axis=-1)
        paired_fakeAB = tf.concat([self._predictedA,
                                self._predictedB],
                                axis=-1)
        """
        paired_real = tf.concat([fmap_apair,
                                fmap_bpair],
                                axis=-1)
        paired_fakeA = tf.concat([fmap_afake,
                                fmap_bpair],
                                axis=-1)
        paired_fakeB = tf.concat([fmap_apair,
                                fmap_bfake],
                                axis=-1)
        paired_fakeAB = tf.concat([fmap_afake,
                                fmap_bfake],
                                axis=-1)
        paired_fake = tf.concat([paired_fakeA,
                                paired_fakeB,
                                paired_fakeAB],axis=0)
        self._real_discrimPaired = self.d_pair(paired_real)
        self._fake_discrimPaired = self.d_pair(paired_fake)


    def _create_loss(self,*args,**kwargs):
        super(SemiAdverCycleGAN,self)._create_loss(*args,**kwargs)
        self._discrimpaired_loss = (tf.losses.mean_squared_error(predictions=self._real_discrimPaired,
                                    labels=0.9*tf.ones_like(self._real_discrimPaired))
     + tf.losses.mean_squared_error(predictions=self._fake_discrimPaired,
                                    labels=tf.zeros_like(self._fake_discrimPaired)))
        self._adver_loss = tf.losses.mean_squared_error(predictions=self._fake_discrimPaired,
                                    labels=0.9*tf.ones_like(self._fake_discrimPaired))
        self._genA_loss += self._ALPHA*self._adver_loss
        self._genB_loss += self._ALPHA*self._adver_loss

    def _create_summary(self,*args,**kwargs):
        super(SemiAdverCycleGAN,self)._create_summary(*args,**kwargs)
        variables = [self._adver_loss,
                    self._discrimpaired_loss,
                    self._xphA_paired,
                    self._xphB_paired]
        types = ['scalar']*2+['image']*2
        names = ['loss/entropy','loss/discrim_paired','image/x_phA_paired','image/x_phB_paired']

        self._summary_pair_op,self._weights_pair_op = learning.create_summary(variables,types,names)

    def _create_optimiser(self,*args,**kwargs):
        super(SemiAdverCycleGAN,self)._create_optimiser(*args,**kwargs)
        discrimpair_solver = tf.contrib.layers.optimize_loss(self._discrimpaired_loss,
                                        self._epoch,
                                         tf.convert_to_tensor(self._lr_variable),
                                        'Adam',
                                        variables=self.d_pair.trainable_weights + self.d_A.trainable_weights+self.d_B.trainable_weights,
                                        increment_global_step=False,)
        with tf.control_dependencies(
            [discrimpair_solver]):
            self._discrimpair_solver = tf.no_op(name='optimisers')

    def train_step(self,A,B,
                    A_paired,B_paired,
                    write_summary=False):
        data={self._xphA: A,
              self._xphB: B,
              self._xphA_paired: A_paired,
              self._xphB_paired: B_paired,
              K.learning_phase():True,}
        (summary,summary_pair,
            global_step,_,_,_) =self.sess.run([
                                    self._summary_op,
                                    self._summary_pair_op,
                                    self._batch_step,
                                    self._solver,
                                    self._discrimpair_solver,
                                    self._batch_step_inc],
                                    feed_dict=data)
        if write_summary:
            self._train_writer.add_summary(summary, global_step)
            self._train_writer.add_summary(summary_pair, global_step)
        return global_step

    def __str__(self):
        return "entropy"
