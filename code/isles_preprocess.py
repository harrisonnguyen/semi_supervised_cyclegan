import numpy as np
import tensorflow as tf
import click
import os
import nibabel as nib
from preprocess import write_example
config = dict()
config["all_modalities"] = ["4DPWI", "ADC", "MTT", "rCBF","rCBV","Tmax","TTP","OT"]
config["training_modalities"] = ["ADC", "MTT", "rCBF","rCBV","Tmax","TTP"]

def itensity_normalize_one_volume(volume,feature_range=(-1.0,1.0)):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    pixels = volume[volume > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    #out_random = np.random.normal(0, 1, size = volume.shape)
    #out[volume == 0] = out_random[volume == 0]

    volume_std = (volume - volume.min()) / float(volume.max() - volume.min())
    return volume_std*(feature_range[1]-feature_range[0]) + feature_range[0]

def resize_by_axis(image, dim_1, dim_2, ax, is_grayscale):

    resized_list = []

    if is_grayscale:
        unstack_img_depth_list = [tf.expand_dims(x,2) for x in tf.unstack(image, axis = ax)]
        for i in unstack_img_depth_list:
            resized_list.append(tf.image.resize_images(i, [dim_1, dim_2],method=0))
        stack_img = tf.squeeze(tf.stack(resized_list, axis=ax))
        #print(stack_img.get_shape())

    else:
        unstack_img_depth_list = tf.unstack(image, axis = ax)
        for i in unstack_img_depth_list:
            resized_list.append(tf.image.resize_images(i, [dim_1, dim_2],method=0))
        stack_img = tf.stack(resized_list, axis=ax)

    return stack_img

def slice_image(img_data,patch_size=128):
    img_data = np.rot90(img_data,3)
    sliced = img_data[:,~np.all(img_data == 0, axis=(0,2)),:]
    sliced = sliced[~np.all(sliced == 0, axis=(1,2)),:,:]
    sliced = sliced[:,:,~np.all(sliced == 0, axis=(0,1))]
    #resize the image to 128,x,128
    x = tf.placeholder(tf.float32,sliced.shape)
   # resized_along_depth = resize_by_axis(x,patch_size,sliced.shape[1],2, True)
   # resized_along_width = resize_by_axis(resized_along_depth,patch_size,patch_size,1,True)
    resized_along_width = resize_by_axis(x,patch_size,patch_size,2,True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        resized = sess.run(resized_along_width,feed_dict={x:sliced})
    # normalise the image
    sliced = itensity_normalize_one_volume(resized)
    return sliced, sliced.shape

def fetch_training_data_files(data_dir):
    dirs = [os.path.join(data_dir,'training/')]
    data_files = []
    for root in dirs:
        for path, subdirs, files in os.walk(root):
            for name in files:
                if '.nii' in name and any(ele in name for ele in config["training_modalities"]) and "__MACOSX" not in path:
                    data_files.append(os.path.join(path, name))
    return data_files

@click.command()
@click.option('--data-dir',
            default="/media/harrison/ShortTerm/Users/HarrisonG/research/isles/",
            type=click.Path(
                    file_okay=False,
                    dir_okay=True,
                    writable=False),
             help="directory to save results",
             show_default=True)

@click.option('--out-dir',
            default="/media/harrison/ShortTerm/Users/HarrisonG/research/semi_supervised_cyclegan/data/isles",
            type=click.Path(
                    file_okay=False,
                    dir_okay=True,
                    writable=False),
             help="directory to save results",
             show_default=True)

def main(data_dir,out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    files = fetch_training_data_files(data_dir)
    for i in files:
        # get the patient number
        patient_id = i.split('/')[9].split('_')[1]

        inputImage = nib.load(i)
        img_data = inputImage.get_data()


        normed_image,shape = slice_image(img_data)
        image = np.transpose(normed_image,(2,0,1))[1:]

        # pad images to 40 slicrs
        image = np.pad(image,((0,40-image.shape[0]),(0,0),(0,0)))
        image = image[:,:,:,np.newaxis]
        shape = image.shape
        print(shape)
        modality = i.split('.')[4].split("_")[-1]
        file_name = "isles-{}-{}".format(patient_id,modality)
        write_example(image.reshape((-1)),out_dir,file_name)

if __name__ == "__main__":
    exit(main())
