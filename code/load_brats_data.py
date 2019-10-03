import tensorflow as tf
import os
import glob
from load_mri_data import _parse_image_function,remove_zeros
import pandas as pd

modality_types = ["t1","t2","t1ce","flair","truth"]
def augment(image):
    image = image['img']

    image = tf.transpose(image,[2,0,1,3])
    #paddings = tf.constant([[0,0,],[8,8,],[8,8],[0,0]])
    #image = tf.pad(image,paddings,"CONSTANT")
    # remove any slices that contain 0

    image = remove_zeros(image)
    # normalise between -1 and 1

    image = image/tf.reduce_max(image)

    image = 2.0*image-1.0
    return image

def map_dataset(file,image_size,buffer_size,shuffle):

    dataset = tf.data.TFRecordDataset(file)
    dataset = dataset.map(lambda x:_parse_image_function(x,image_size))
    dataset = dataset.map(lambda x:augment(x))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    return dataset

def load_data(fileA,
                fileB,
                image_size,
                buffer_size=30,
                shuffle=True,
                repeat=1):
    datasetA = map_dataset(fileA,image_size,buffer_size,shuffle)
    datasetB = map_dataset(fileB,image_size,buffer_size,shuffle)
    dataset =  tf.data.Dataset.zip((datasetA, datasetB))
    dataset= dataset.repeat(repeat)
    return dataset

def get_files(dir,modality):
    if modality not in modality_types:
        raise ValueError("Invalid sim type. Expected one of: %s" % modality_types)
    return glob.glob(os.path.join(dir,"*-{}.tfrecords".format(modality)))

def get_data_split(data_dir,modality,set_choice,include_pair=True,split_filename="data/brats_files.csv",):
    set_choices = ["setA","setB","pair","test"]
    if modality not in modality_types:
        raise ValueError("Invalid sim type. Expected one of: %s" % modality_types)
    if set_choice not in set_choices:
        raise ValueError("Invalid sim type. Expected one of: %s" % set_choices)
    df = pd.read_csv(split_filename)
    set_df = data_dir+ df[df["Set"] == set_choice]["Filename"].astype(str) + "-"+modality+".tfrecords"
    set_files = list(set_df.values)

    if include_pair:
        pair_df =data_dir+ df[df["Set"] == "pair"]["Filename"].astype(str) + "-"+modality+".tfrecords"
        set_files += list(pair_df.values)
    return set_files

def get_generator(data_dir,image_size,mod_a,mod_b,include_pair=False,
                    split_filename="data/brats_files.csv",buffer_size=10):
    generator_dict = {}
    setA_files = get_data_split(data_dir,mod_a,"setA",include_pair=False,
    split_filename=split_filename)[:2]
    setB_files = get_data_split(data_dir,mod_b,"setB",include_pair=False,
    split_filename=split_filename)[:2]
    training = load_data(setA_files,
                        setB_files,
                        image_size=image_size,
                        buffer_size=10)
    generator_dict["training"] = training
    if include_pair:
        pairA_files = get_data_split(
                        data_dir,
                        mod_a,
                        "pair",
                        include_pair=True,
                        split_filename=split_filename)
        pairB_files = get_data_split(
                        data_dir,
                        mod_b,
                        "pair",
                        include_pair=True,
                        split_filename="data/brats_files.csv")
        pair_training = load_data(pairA_files,
                            pairB_files,
                            image_size=image_size,
                            buffer_size=10,
                            repeat=None)
        generator_dict["pair"] = pair_training
    valA_files = get_data_split(data_dir,mod_a,"test",split_filename)
    valB_files = get_data_split(data_dir,mod_b,"test",split_filename)
    val = load_data(valA_files[0],
                    valB_files[0],
                        image_size=image_size,
                        buffer_size=1,
                        shuffle=False,
                        repeat=None)
    generator_dict["val"] = val
    return generator_dict

def main():
    dir = "data/brats2018/"
    files = get_files(dir,"t1")
    dataset = load_data(files,
                    image_size=[240,240,155,1])

    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    next_image = iterator.get_next()
    sess = tf.compat.v1.Session()
    while True:
        try:
            image = sess.run(next_image)
        except tf.errors.OutOfRangeError:
            break
if __name__ == "__main__":
    main()
