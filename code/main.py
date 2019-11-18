import tensorflow as tf
import pandas as pd
from model.cyclegan import CycleGAN
from model.semiadversarialcycle import SemiAdverCycleGAN
from model.cwrgan import SemiWassersteinCycleGAN
from load_brats_data import modality_types,get_generator
from isles_preprocess import config
import click
import numpy as np
import csv
import os
import random
import string
import yaml
import pandas as pd
def index_gen(n_slices,batch_size):
    index = np.array(range(n_slices))
    np.random.shuffle(index)
    for i in range(int(np.floor(n_slices/batch_size))):
        items = index[i*batch_size:(i+1)*batch_size]
        yield items
def write_params(params,checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    w = csv.writer(open(
                    os.path.join(checkpoint_dir,
                                "config.csv"),
                    "w+"))
    for key, val in params.items():
        w.writerow([key, val])
    print_settings(params)
    return params

def print_settings(params):
    click.secho(
        "Summary of all parameter settings:\n"
        "----------------------------------\n"
        "{}".format(yaml.dump(params, default_flow_style=False)),
        fg="yellow",
    )
    return params

def generate_experiment_id(length=6):
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(length))

def train_semi(gan,i,next_training,next_training_pair,batch_size,summary_freq):
    img_A,img_B = gan.sess.run(next_training)
    img_pairA,img_pairB = gan.sess.run(next_training_pair)

    index = np.array(range(0,min(img_pairA.shape[0],img_pairB.shape[0])))
    gen = index_gen(min(img_A.shape[0],img_B.shape[0]),batch_size)
    for ele in gen:
        samples = np.random.randint(0,len(index),batch_size)
        write_summary = i%summary_freq == 0
        epoch = gan.train_step(
                    img_A[ele],
                    img_B[ele],
                    img_pairA[index[samples]],
                    img_pairB[index[samples]],
                    write_summary=write_summary)
        i+=1
    return i

def train_cycle(gan,i,next_training,batch_size,summary_freq):
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

def validate(gan,iterator_val,next_val,batch_size):
    gan.sess.run(iterator_val.initializer)
    img_A,img_B = gan.sess.run(next_val)
    index = np.array(range(min(img_A.shape[0],img_B.shape[0])))
    np.random.shuffle(index)
    #n_slice = np.min(batch_size*20,len(index))
    values = index[:batch_size*5]
    print(values)

    gan.validate(
        img_A[values],
        img_B[values])

def score(gan,iterator,next_set,batch_size,method="mse"):
    gan.sess.run(iterator.initializer)
    k = 0
    total_loss = 0
    while True:
        try:
            img_A,img_B = gan.sess.run(next_set)
            index = np.array(range(img_A.shape[0]))
            for i in range(int(np.floor(img_A.shape[0]/batch_size))):
                items = index[i*batch_size:(i+1)*batch_size]
                total_loss += gan.score(img_A[items],img_B[items],method)
                k+=1
        except tf.errors.OutOfRangeError:
            break
    return total_loss/k

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
            type=click.Choice(modality_types+config["training_modalities"]),
             help="Choice of modality",
             show_default=True)
@click.option('--mod-b',
            type=click.Choice(modality_types+config["training_modalities"]),
             help="Choice of modality",
             show_default=True)
@click.option('--model',
            default="semi",
            type=click.Choice(["cycle","semi","wasser"]),
             help="Choice of model type",
             show_default=True)
@click.option('--dataset',
            default="brats",
            type=click.Choice(["brats","isles"]),
             help="Choice of model type",
             show_default=True)
@click.option('--experiment-id',
            default=0,
            type=click.INT,
             help="Experiment id",
             show_default=True)
@click.option('--semi-loss-weight',
            default=1.0,
            type=click.FLOAT,
             help="Relative loss of cycle",
             show_default=True)
@click.option('--only-pair',
            is_flag=True,
             help="Flag to use only paired data",
             show_default=True)
def main(checkpoint_dir,
        data_dir,
        gf,df,depth,n_channels,
        cycle_loss_weight,learning_rate,batch_size,n_epochs,
        summary_freq,end_learning_rate,begin_decay,decay_steps,mod_a,mod_b,
        model,dataset,experiment_id,semi_loss_weight,only_pair):
    params = click.get_current_context().params
    checkpoint_dir= os.path.join(checkpoint_dir,dataset)
    checkpoint_dir = os.path.join(checkpoint_dir,"{}_{}".format(mod_a,mod_b))
    checkpoint_dir= os.path.join(checkpoint_dir,model)
    checkpoint_dir= os.path.join(
                    checkpoint_dir,
                    "gf{}_df{}_depth{}_lambda{}_alpha{}".format(
                    gf,df,depth,cycle_loss_weight,semi_loss_weight))
    checkpoint_dir= os.path.join(checkpoint_dir,"experiment{}".format(experiment_id))
    write_params(params,checkpoint_dir)

    # load the results file
    try:
        results_df = pd.read_csv(os.path.join(checkpoint_dir,"results.csv"))
    except:
        results_df = pd.DataFrame(columns=["epoch","val_mse","val_mae","test_mse","test_mae"])

    if dataset == "brats":
        image_size = [240,240,155,1]
        patch_size = 240
        split_filename = "data/brats_files.csv"
    elif dataset == "isles":
        image_size = [40,128,128,1]
        patch_size = 128
        split_filename = "data/isles_files.csv"
    if model == "semi":
        gan = SemiAdverCycleGAN(
                base_dir=checkpoint_dir,
                gf=gf,
                df=df,
                depth=depth,
                patch_size=patch_size,
                n_modality=n_channels,
                cycle_loss_weight=cycle_loss_weight,
                initial_learning_rate=learning_rate,
                begin_decay=begin_decay,
                end_learning_rate=end_learning_rate,
                decay_steps=decay_steps,
                entropy_weight=semi_loss_weight)
        include_pair = True
    elif model == "cycle":
        gan = CycleGAN(
                base_dir=checkpoint_dir,
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
                base_dir=checkpoint_dir,
                gf=gf,
                df=df,
                depth=depth,
                patch_size=patch_size,
                n_modality=n_channels,
                cycle_loss_weight=cycle_loss_weight,
                initial_learning_rate=learning_rate,
                begin_decay=begin_decay,
                end_learning_rate=end_learning_rate,
                decay_steps=decay_steps,
                mse_weight=semi_loss_weight)
        include_pair = True
    dataset_gen = get_generator(
                    data_dir,
                    image_size,mod_a,mod_b,
                    include_pair=include_pair,
                    split_filename=split_filename,
                    dataset_type=dataset,
                    only_pair=only_pair)
    #set_A,set_B = get_data_split(data_dir,mod_a,mod_b)
    sess = gan.sess
    iterator_training = tf.compat.v1.data.make_initializable_iterator(dataset_gen["training"])
    next_training = iterator_training.get_next()
    iterator_val = tf.compat.v1.data.make_initializable_iterator(dataset_gen["val"])
    next_val = iterator_val.get_next()
    iterator_test = tf.compat.v1.data.make_initializable_iterator(dataset_gen["test"])
    next_test = iterator_test.get_next()
    #sess.run(iterator_val.initializer)
    if include_pair:
        iterator_training_pair = tf.compat.v1.data.make_initializable_iterator(dataset_gen["pair"])
        next_training_pair = iterator_training_pair.get_next()
        sess.run(iterator_training_pair.initializer)

    # load existing model
    i = gan.restore_latest_checkpoint()
    best_val_loss = np.inf
    recent_val_loss = []
    # begin training
    for k in range(n_epochs):
        sess.run(iterator_training.initializer)
        while True:
            try:
                if include_pair:
                    i = train_semi(gan,i,next_training,next_training_pair,batch_size,summary_freq)
                else:
                    i = train_cycle(gan,i,next_training,batch_size,summary_freq)
            except tf.errors.OutOfRangeError:
                gan.save_checkpoint()
                current_epoch = gan.increment_epoch()
                print("finished epoch %d. Saving checkpoint" %current_epoch)
                # run validation
                validate(gan,iterator_val,next_val,batch_size)

                # write results to file
                if k%2 == 0:
                    results = {}
                    results["epoch"] = current_epoch - 1
                    val_score = score(gan,iterator_val,next_val,batch_size)
                    print("Val mse: {:.04f}".format(val_score))
                    results["val_mse"] = val_score

                    val_score_mae = score(gan,iterator_val,next_val,batch_size,"mae")
                    print("Val mae: {:.04f}".format(val_score_mae))
                    results["val_mae"] = val_score_mae

                    test_score = score(gan,iterator_test,next_test,batch_size)
                    results["test_mse"] = test_score
                    print("test_mse: {:.04f}".format(test_score))

                    test_score = score(gan,iterator_test,next_test,batch_size,"mae")
                    results["test_mae"] = test_score
                    print("test_mae: {:.04f}".format(test_score))
                    results_df = results_df.append(results,ignore_index=True)
                    results_df.to_csv(os.path.join(checkpoint_dir,"results.csv"),index=False)
                    recent_val_loss.append(val_score)
                    if val_score < best_val_loss:
                        best_val_loss = val_score
                        gan.save_checkpoint(save_best=True)
                    #elif np.mean(recent_val_loss[-10:]) > best_val_loss and len(recent_val_loss) > 10:
                    #    new_lr = gan.update_learning_rate(fraction=0.8)
                    #    print("reducing lr to {:0.6f}".format(new_lr))
                break
    results = {}
    results["epoch"] = n_epochs
    val_score = score(gan,iterator_val,next_val,batch_size)
    print("Val mse: {:.04f}".format(val_score))
    results["val_mse"] = val_score
    val_score = score(gan,iterator_val,next_val,batch_size,"mae")
    print("Val mae: {:.04f}".format(val_score))
    results["val_mae"] = val_score
    test_score = score(gan,iterator_test,next_test,batch_size)
    print("test_mse: {:.04f}".format(test_score))
    results["test_mse"] = test_score
    test_score = score(gan,iterator_test,next_test,batch_size,"mae")
    print("test_mae: {:.04f}".format(test_score))
    results["test_mae"] = test_score

    results_df = results_df.append(results,ignore_index=True)
    results_df.to_csv(os.path.join(checkpoint_dir,"final.csv"),index=False)

if __name__ == "__main__":
    main()
