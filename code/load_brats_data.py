import tensorflow as tf
import os
import glob
from load_mri_data import _parse_image_function,remove_zeros

modality_types = ["t1","t2","t1ce","flair","truth"]
def augment(image):
    image = image['img']

    image = tf.transpose(image,[2,0,1,3])
    paddings = tf.constant([[0,0,],[8,8,],[8,8],[0,0]])
    image = tf.pad(image,paddings,"CONSTANT")
    # remove any slices that contain 0

    image = remove_zeros(image)
    # normalise between -1 and 1

    image = image/tf.reduce_max(image)

    image = 2.0*image-1.0
    return image


def load_data(filenames,
                image_size,
                buffer_size=30):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda x:_parse_image_function(x,image_size))
    dataset = dataset.map(lambda x:augment(x))
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset= dataset.repeat(1)
    return dataset

def get_files(dir,modality):
    if modality not in modality_types:
        raise ValueError("Invalid sim type. Expected one of: %s" % modality_types)
    return glob.glob(os.path.join(dir,"*-{}.tfrecords".format(modality)))

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
