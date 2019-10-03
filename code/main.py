import tensorflow as tf
import pandas as pd
from model.cyclegan import CycleGAN
from model.semiadversarialcycle import SemiAdverCycleGAN
from model.cwrgan import SemiWassersteinCycleGAN
from load_brats_data import modality_types,get_generator
import click
import numpy as np
import csv
import os
def index_gen(n_slices,batch_size):
    index = np.array(range(n_slices))
    np.random.shuffle(index)
    for i in range(int(np.floor(n_slices/batch_size))):
        items = index[i*batch_size:(i+1)*batch_size]
        yield items
def write_params(params):
    if not os.path.exists(params["checkpoint_dir"]):
        os.makedirs(params["checkpoint_dir"])
    w = csv.writer(open(
                    os.path.join(params["checkpoint_dir"],
                                "config.csv"),
                    "w+"))
    for key, val in params.items():
        w.writerow([key, val])
    return

def train_semi(gan,i,next_training,next_training_pair):
    img_A,img_B = gan.sess.run(next_training)
    img_pairA,img_pairB = gan.sess.run(next_training_pair)

    index = np.array(range(0,min(img_pairA.shape[0],img_pairA.shape[0])))
    gen = index_gen(min(img_A.shape[0],img_B.shape[0]),batch_size)
    for ele in gen:
        np.random.shuffle(index)
        write_summary = i%summary_freq == 0
        epoch = gan.train_step(
                    img_A[ele],
                    img_B[ele],
                    img_pairA[index[:batch_size]],
                    img_pairB[index[:batch_size]],
                    write_summary=write_summary)
        i+=1
    return i

def train_cycle(gan,i,next_training):
    img_A,img_B = gan.sess.run(next_training)
    gen = index_gen(min(img_A.shape[0],img_B.shape[0]),batch_size)
    for ele in gen:
        write_summary = i%summary_freq == 0
        epoch = gan.train_step(
                    img_A[ele],
                    img_B[ele],
                    write_summary=write_summary)
    i+=1
    return i

def validate(gan,next_val,batch_size):
    gan.save_checkpoint()
    img_A,img_B = sess.run(next_val)
    index = np.array(range(10,min(img_A.shape[0],img_B.shape[0])-20))
    np.random.shuffle(index)
    values = index[:batch_size*4]
    gan.validate(
        img_A[values],
        img_B[values])
    current_epoch = gan.increment_epoch()
    print("finished epoch %d. Saving checkpoint" %current_epoch)
@click.command()
@click.option('--checkpoint-dir',
            type=click.Path(
                    file_okay=False,
                    dir_okay=True,
                    writable=True),
             help="directory to save results",
             show_default=True)
@click.option('--data-dir',
            default="data/brats2018/",
            type=click.Path(
                    file_okay=False,
                    dir_okay=True,
                    writable=False),
             help="directory of data",
             show_default=True)
@click.option('--gf',
            default=32,
            type=click.INT,
             help="Number of initial filters in generator",
             show_default=True)
@click.option('--df',
            default=64,
            type=click.INT,
             help="Number of initial filters in discriminator",
             show_default=True)
@click.option('--depth',
            default=3,
            type=click.INT,
             help="Number of convolution blocks for discirm and gen",
             show_default=True)
@click.option('--patch-size',
            default=256,
            type=click.INT,
             help="Size of image",
             show_default=True)
@click.option('--n-channels',
            default=1,
            type=click.INT,
             help="Number of channels",
             show_default=True)
@click.option('--cycle-loss-weight',
            default=10.0,
            type=click.FLOAT,
             help="Relative loss of cycle",
             show_default=True)
@click.option('--learning-rate',
            default=2e-4,
            type=click.FLOAT,
             help="Initial learning rate",
             show_default=True)
@click.option('--batch-size',
            default=16,
            type=click.INT,
             help="batch size for training",
             show_default=True)
@click.option('--n-epochs',
            default=20,
            type=click.INT,
             help="enumber of epochs for training",
             show_default=True)
@click.option('--summary-freq',
            default=500,
            type=click.INT,
             help="enumber of epochs for training",
             show_default=True)
@click.option('--end-learning-rate',
            default=2e-6,
            type=click.FLOAT,
             help="ending learning rate to decay to",
             show_default=True)
@click.option('--begin-decay',
            default=100,
            type=click.INT,
             help="epochs to begin decay",
             show_default=True)
@click.option('--decay-steps',
            default=100,
            type=click.INT,
             help="number of epochs to decay learning rate",
             show_default=True)
@click.option('--mod-a',
            type=click.Choice(modality_types),
             help="Choice of brats modality",
             show_default=True)
@click.option('--mod-b',
            type=click.Choice(modality_types),
             help="Choice of brats modality",
             show_default=True)
@click.option('--model',
            default="semi",
            type=click.Choice(["cycle","semi","wasser"]),
             help="Choice of model type",
             show_default=True)
@click.option('--dataset',
            default="brats",
            type=click.Choice(["brats"]),
             help="Choice of model type",
             show_default=True)
def main(checkpoint_dir,
        data_dir,
        gf,df,depth,patch_size,n_channels,
        cycle_loss_weight,learning_rate,batch_size,n_epochs,
        summary_freq,end_learning_rate,begin_decay,decay_steps,mod_a,mod_b,
        model,dataset):
    params = click.get_current_context().params
    write_params(params)
    if dataset == "brats":
        image_size = [240,240,155,1]
    if model == "semi":
        gan = SemiAdverCycleGAN(
                base_dir=os.path.join(checkpoint_dir,model),
                gf=gf,
                df=df,
                depth=depth,
                patch_size=patch_size,
                n_modality=n_channels,
                cycle_loss_weight=cycle_loss_weight,
                initial_learning_rate=learning_rate,
                begin_decay=begin_decay,
                end_learning_rate=end_learning_rate,
                decay_steps=decay_steps)
        include_pair = True
    elif model == "cycle":
        gan = CycleGAN(
                base_dir=os.path.join(checkpoint_dir,model),
                gf=gf,
                df=df,
                depth=depth,
                patch_size=patch_size,
                n_modality=n_channels,
                cycle_loss_weight=cycle_loss_weight,
                initial_learning_rate=learning_rate,
                begin_decay=begin_decay,
                end_learning_rate=end_learning_rate,
                decay_steps=decay_steps)
        include_pair = False
    elif model == "wasser":
        gan = SemiWassersteinCycleGAN(
                base_dir=os.path.join(checkpoint_dir,model),
                gf=gf,
                df=df,
                depth=depth,
                patch_size=patch_size,
                n_modality=n_channels,
                cycle_loss_weight=cycle_loss_weight,
                initial_learning_rate=learning_rate,
                begin_decay=begin_decay,
                end_learning_rate=end_learning_rate,
                decay_steps=decay_steps)
        include_pair = True
    dataset_gen = get_generator(
                    data_dir,
                    image_size,mod_a,mod_b,
                    include_pair=include_pair,
                        )
    #set_A,set_B = get_data_split(data_dir,mod_a,mod_b)
    """
    setA_files = get_data_split(data_dir,mod_a,"setA",include_pair=False,
    split_filename="data/brats_files.csv")
    setB_files = get_data_split(data_dir,mod_b,"setB",include_pair=False,
    split_filename="data/brats_files.csv")
    pairA_files = get_data_split(
                    data_dir,
                    mod_a,
                    "pair",
                    include_pair=True,
                    split_filename="data/brats_files.csv")
    pairB_files = get_data_split(
                    data_dir,
                    mod_b,
                    "pair",
                    include_pair=True,
                    split_filename="data/brats_files.csv")
    valA_files = get_data_split(data_dir,mod_a,"test","data/brats_files.csv")
    valB_files = get_data_split(data_dir,mod_b,"test","data/brats_files.csv")
    training = load_data(setA_files,
                        setB_files,
                        image_size=image_size,
                        buffer_size=10)
    pair_training = load_data(pairA_files,
                        pairB_files,
                        image_size=image_size,
                        buffer_size=10,
                        repeat=None)
    val = load_data(valA_files[0],
                    valB_files[0],
                        image_size=image_size,
                        buffer_size=1,
                        shuffle=False,
                        repeat=None)
    """

    sess = gan.sess
    iterator_training = tf.compat.v1.data.make_initializable_iterator(dataset_gen["training"])
    next_training = iterator_training.get_next()
    iterator_val = tf.compat.v1.data.make_initializable_iterator(dataset_gen["val"])
    next_val = iterator_val.get_next()
    sess.run(iterator_val.initializer)
    if include_pair:
        iterator_training_pair = tf.compat.v1.data.make_initializable_iterator(dataset_gen["pair"])
        next_training_pair = iterator_training_pair.get_next()
        sess.run(iterator_training_pair.initializer)
    # load existing model
    try:
        i = gan.restore_latest_checkpoint()
        print("Restoring at step {}".format(i))
    except:
        i = 0
        print("Creating new model")

    # begin training
    for k in range(n_epochs):
        sess.run(iterator_training.initializer)
        while True:
            try:
                if include_pair:
                    i = train_semi(gan,i,next_training,next_training_pair)
                else:
                    i = train_cycle(gan,i,next_training)
            except tf.errors.OutOfRangeError:
                # run validation
                validate(gan,next_val,batch_size)
                break

if __name__ == "__main__":
    main()
