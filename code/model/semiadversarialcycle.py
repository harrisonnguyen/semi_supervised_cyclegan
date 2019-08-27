from cyclegan import CycleGAN

class SemiAdverCycleGAN(CycleGAN):
    def __init__(self,entropy_weight,*args,**kwargs):
        self._ALPHA = entropy_weight
        super(SemiAdverCycleGAN,self).__init__(*args,**kwargs)


    def _build_graph(self,*args,**kwargs):
        super(SemiAdverCycleGAN,self)._build_graph(*args,**kwargs)

        self._xphA_paired = tf.placeholder(tf.float32, [None,patch_size,patch_size,n_modality])
        self._xphB_paired = tf.placeholder(tf.float32, [None,patch_size,patch_size,n_modality])

        # we concatenate the paired images on the channel axis
        img_shape = self.img_shape
        img_shape[-1] = self.img_shape*2
        self.d_pair = discriminator(self.img_shape,df,depth)

        paired_real = tf.concat([self._xphA_paired,
                                self._xphB_paired],
                                axis=-1)
        paired_fakeA = tf.concat([self._predictedA,
                                self._reconstructB],
                                axis=-1)
        paired_fakeB = tf.concat([self._reconstructA,
                                self._predictedB],
                                axis=-1)
        paired_fake = tf.concat([paired_fakeA,
                                paired_fakeB],axis=0)
        self._real_discrimPaired = self.d_pair(paired_real)
        self._fake_discrimPaired = self.d_pair(paired_fake)


    def _create_loss(self,*args,**kwargs):
        super(SemiAdverCycleGAN,self)._create_loss(*args,**kwargs)
        self._discrimpaired_loss = (tf.losses.mean_squared_error(predictions=self._real_discrimPaired,
                                    labels=tf.ones_like(self._real_discrimPaired))
     + tf.losses.mean_squared_error(predictions=self._fake_discrimPaired,
                                    labels=tf.zeros_like(self._fake_discrimPaired)))
        self._adver_loss = tf.losses.mean_squared_error(predictions=self._fake_discrimPaired,
                                    labels=tf.ones_like(self._fake_discrimPaired))

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
                                                 self.initial_learning_rate,
                                                'Adam',
                                                variables=self.d_pair.trainable_weights,
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
