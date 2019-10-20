from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate,LeakyReLU,UpSampling3D, Conv3D, Conv2D, UpSampling2D,Lambda,Add
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D,Conv2DTranspose
from tensorflow.keras.models import Model
from utils.instance_norm import InstanceNormalization


def residual_block(layer_input,filters,f_size=3):
    d = ZeroPadding2D()(layer_input)
    d = Conv2D(filters, kernel_size=f_size, strides=1, padding='valid')(d)
    d = InstanceNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)


    d = ZeroPadding2D()(d)
    d = Conv2D(filters, kernel_size=f_size, strides=1, padding='valid')(d)
    d = InstanceNormalization()(d)
    output = Add()([layer_input,d])
    return output

def conv2d(layer_input, filters, f_size=3,normalise=True):
    """Layers used during downsampling"""
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)

    if normalise:
        d = InstanceNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    return d

def deconv2d(layer_input, filters, skip_input=None, f_size=3, dropout_rate=0,
activation='relu'):
    """Layers used during upsampling"""
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(u)
    #u = Conv2DTranspose(filters,f_size,strides=2, padding='same')(layer_input)

    if dropout_rate:
        u = Dropout(dropout_rate)(u)
    u = InstanceNormalization()(u)
    u = LeakyReLU(alpha=0.2)(u)
    if skip_input is not None:
        u = Concatenate()([u, skip_input])
    return u

def resnet_gen(input_img_shape,gf,depth,n_res_block=3):
    d0 = Input(shape=input_img_shape)
    input = d0

    # down sample
    for i in range(depth):
        if i==0:
            normalise=False
        else:
            normalise=True
        output = conv2d(input,gf*(2**i),normalise=normalise)
        input = output

    # residual block
    for i in range(n_res_block):
        output = residual_block(input,gf*(2**(depth-1)))
        input = output

    # upsample
    for i in range(depth-1, -1, -1):
        output = deconv2d(input,gf*2**i)
        input = output
    #u4 = UpSampling2D(size=2)(input)
    #output_img = Conv2DTranspose(input_img_shape[-1], kernel_size=3, strides=2, padding='same', activation='tanh')(input)
    output_img = Conv2D(input_img_shape[-1], kernel_size=3, strides=1, padding='same', activation='tanh')(input)

    return Model(d0, output_img)

def generator(input_img_shape,gf,depth):
    """U-Net Generator"""
    # Image input
    d0 = Input(shape=input_img_shape)
    input = d0
    layers = []
    # Downsampling
    for i in range(depth+1):
        if i==0:
            normalise=False
        else:
            normalise=True
        output = conv2d(input,gf*(2**i),normalise=normalise)
        layers.append(output)
        input = output
    for i in range(depth-1, -1, -1):
        output = deconv2d(input,gf*(2**i),layers[i])
        input = output

    output = deconv2d(input,gf)
    #u4 = UpSampling2D(size=2)(input)
    output_img = Conv2D(input_img_shape[-1], kernel_size=3, strides=1, padding='same', activation='tanh')(output)
    #output_img = Conv2DTranspose(gf, kernel_size=3, strides=2, padding='same')(output)
    return Model(d0, output_img)

def d_layer(layer_input, filters, f_size=4, normalization=True):
    """Discriminator layer"""
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)

    if normalization:
        d = InstanceNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    return d

def discriminator(input_img_shape,df,depth):
    img = Input(shape=input_img_shape)

    d1 = d_layer(img, df, normalization=False)
    input = d1
    for i in range(1,depth):
        output = d_layer(input, df*(2**i))
        input = output
    output = d_layer(input, df*(2**depth))
    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(output)

    return Model(img, validity)
