import click
import os
import glob
import nibabel as nib
import numpy as np
import tensorflow as tf

config = dict()
config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
config["training_modalities"] = config["all_modalities"]

def write_example(image,directory,file_name):
    writer = tf.python_io.TFRecordWriter(os.path.join(directory,file_name +".tfrecords"))
    temp_dict = {}
    temp_dict['img'] = tf.train.Feature(float_list=tf.train.FloatList(value = image))

    #construct the example proto object
    example = tf.train.Example(
                features = tf.train.Features(
                                feature = temp_dict))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    writer.write(serialized)
    writer.close()
    return serialized

def fetch_training_data_files(data_dir):
    training_data_files = list()
    for subject_dir in glob.glob(os.path.join(data_dir, "*", "*")):
        subject_files = list()
        for modality in config["training_modalities"] + ["truth"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    return training_data_files


@click.command()
@click.option('--data-dir',
            default="/media/harrison/ShortTerm/Users/HarrisonG/brats_temp_pre",
            type=click.Path(
                    file_okay=False,
                    dir_okay=True,
                    writable=False),
             help="directory to save results",
             show_default=True)

@click.option('--out-dir',
            default="/media/harrison/ShortTerm/Users/HarrisonG/research/semi_supervised_cyclegan/data/brats2018",
            type=click.Path(
                    file_okay=False,
                    dir_okay=True,
                    writable=False),
             help="directory to save results",
             show_default=True)
def main(data_dir,
        out_dir):
    training_files = fetch_training_data_files(data_dir)
    for patients in training_files:
        for file in patients:
            try:
                epi_img = nib.load(file)
                img_data = epi_img.get_fdata()
                split = file.split("/")
                modality = split[-1].split(".")[0]
                file_name = "{}-{}-{}".format(split[-3],split[-2],modality)
                write_example(np.reshape(np.array(img_data),(-1)),out_dir,file_name)
            except FileNotFoundError:
                print(file)
if __name__ == "__main__":
    exit(main())
