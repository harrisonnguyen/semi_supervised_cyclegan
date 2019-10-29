import tensorflow as tf
import os
import glob
from load_mri_data import _parse_image_function,remove_zeros
import pandas as pd

modality_types = ["t1","t2","t1ce","flair","truth"]


def preprocess(image,dataset_type,augment_data=False):
    image = image['img']

    if dataset_type =="brats":
        image = _preprocess(image,augment_data)
    elif dataset_type == "isles":
        image = _preprocess_isles(image,augment_data)
    return image

def _preprocess(image,augment_data):
    image = tf.transpose(image,[2,0,1,3])

    if augment_data:
        image = augment(image)
    # remove any slices that contain 0
    image = remove_zeros(image)

    # normalise between -1 and 1
    image = image/tf.reduce_max(image)
    image = 2.0*image-1.0
    return image

def preprocess_pair(imageA,imageB,dataset_type,augment_data=False):
    imageA = imageA['img']
    imageB = imageB['img']
    image = tf.concat([imageA,imageB],axis=-1)
    if dataset_type == "brats":
        image = _preprocess(image,augment_data)
    elif dataset_type == "isles":
        image = _preprocess_isles(image,augment_data)
    return tf.expand_dims(image[:,:,:,0],-1),tf.expand_dims(image[:,:,:,1],-1)

def augment(image,dim_reduce=(60,20)):
    # perform random cropping
    shape = image.get_shape().as_list()
    image = tf.image.random_crop(
                image,
                [shape[0],shape[1]-dim_reduce[0],shape[2]-dim_reduce[1],shape[-1]],)
    image = tf.image.pad_to_bounding_box(
                image,
                int(dim_reduce[0]/2),
                int(dim_reduce[1]/2),
                shape[1],
                shape[2])
    return image

def _preprocess_isles(image,augment_data):
    image = image[1:]
    #if augment_data:
    #    image = augment(image,(18,18))
    # remove any slices that contain 0
    image = remove_zeros(image)
    return image


def map_dataset(file,image_size,buffer_size,shuffle,dataset_type,augment_data=False):
    dataset = tf.data.TFRecordDataset(file)
    dataset = dataset.map(lambda x:_parse_image_function(x,image_size))
    dataset = dataset.map(lambda x:preprocess(
                                    x,
                                    dataset_type=dataset_type,
                                    augment_data=augment_data))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    return dataset

def load_data(fileA,
                fileB,
                image_size,
                dataset_type,
                buffer_size=30,
                shuffle=True,
                repeat=1,
                augment=False,
                ):
    datasetA = map_dataset(fileA,image_size,buffer_size,shuffle,dataset_type=dataset_type,augment_data=augment)
    datasetB = map_dataset(fileB,image_size,buffer_size,shuffle,dataset_type=dataset_type,augment_data=augment)
    dataset =  tf.data.Dataset.zip((datasetA, datasetB))
    dataset= dataset.repeat(repeat)
    return dataset

def load_data_pair(fileA,
                fileB,
                image_size,
                dataset_type,
                buffer_size=30,
                shuffle=True,
                repeat=1,
                augment=False):
    datasetA = tf.data.TFRecordDataset(fileA)
    datasetB = tf.data.TFRecordDataset(fileB)
    datasetA = datasetA.map(lambda x:_parse_image_function(x,image_size))
    datasetB = datasetB.map(lambda x:_parse_image_function(x,image_size))
    dataset =  tf.data.Dataset.zip((datasetA, datasetB))
    dataset = dataset.map(lambda x,y:preprocess_pair(x,y,dataset_type,augment_data=augment))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset= dataset.repeat(repeat)
    return dataset

def get_files(dir,modality):
    if modality not in modality_types:
        raise ValueError("Invalid sim type. Expected one of: %s" % modality_types)
    return glob.glob(os.path.join(dir,"*-{}.tfrecords".format(modality)))

def get_data_split(data_dir,modality,set_choice,include_pair=True,split_filename="data/brats_files.csv",):
    set_choices = ["setA","setB","pair","test"]
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
                    split_filename="data/brats_files.csv",buffer_size=10,dataset_type="brats"):
    generator_dict = {}
    setA_files = get_data_split(data_dir,mod_a,"setA",include_pair=False,
    split_filename=split_filename)
    setB_files = get_data_split(data_dir,mod_b,"setB",include_pair=False,
    split_filename=split_filename)
    training = load_data(setA_files,
                        setB_files,
                        image_size=image_size,
                        buffer_size=5,
                        augment=True,
                        dataset_type=dataset_type)
    generator_dict["training"] = training
    if include_pair:
        pairA_files = get_data_split(
                        data_dir,
                        mod_a,
                        "pair",
                        include_pair=True,
                        split_filename=split_filename,)
        pairB_files = get_data_split(
                        data_dir,
                        mod_b,
                        "pair",
                        include_pair=True,
                        split_filename=split_filename)
        pair_training = load_data_pair(pairA_files,
                            pairB_files,
                            image_size=image_size,
                            buffer_size=5,
                            repeat=None,
                            augment=True,
                            dataset_type=dataset_type)
        generator_dict["pair"] = pair_training
    valA_files = get_data_split(data_dir,mod_a,"test",split_filename=split_filename)
    valB_files = get_data_split(data_dir,mod_b,"test",split_filename=split_filename)
    val = load_data_pair(valA_files[0],
                    valB_files[0],
                        image_size=image_size,
                        buffer_size=1,
                        shuffle=False,
                        repeat=1,
                        dataset_type=dataset_type)
    test = load_data_pair(valA_files[1:],
                    valB_files[1:],
                        image_size=image_size,
                        buffer_size=1,
                        shuffle=False,
                        repeat=1,
                        dataset_type=dataset_type)
    generator_dict["val"] = val
    generator_dict["test"] = test
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
