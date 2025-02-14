import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import Add, Concatenate, Activation, Dropout, LeakyReLU, MaxPooling2D


def BasicEncoder(D):
    """
    The BasicEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.
    Input: (N, 3, H, W), (N, H, W, D)
    Output: (N, 3, H, W)
    """
    Cover = Input(shape=(None, None, 3), name=f'cover_image') # (N, H, W, 3)
    Message = Input(shape=(None, None, D), name=f'message_data') # (N, H, W, D)

    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv')(Cover) # (N, H, W, 3) -> (N, H, W, 32)
    a = BatchNormalization(name='a_normalize')(a)

    b_concatenate = Concatenate(name='b_concatenate')([a, Message]) # (N, H, W, 32) + (N, H, W, D) -> (N, H, W, 32 + D)
    b = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='b_conv')(b_concatenate) # (N, H, W, 32 + D) -> (N, H, W, 32)
    b = BatchNormalization(name='b_normalize')(b)

    c = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='c_conv')(b) # (N, H, W, 32) -> (N, H, W, 32)
    c = BatchNormalization(name='c_normalize')(c)

    Encoder_d = Conv2D(3, kernel_size=3, padding='same', activation='tanh', name='Encoder_conv_tanh')(c) # (N, H, W, 32) -> (N, H, W, 3)
    
    model = Model(inputs=[Cover, Message], outputs=Encoder_d, name='ResnetSteganoGAN_basic_encoder')
    return model

def ResidualEncoder(D):
    """
    The ResidualEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.
    Input: (N, H, W, 3), (N, H, W, D)
    Output: (N, H, W, 3)
    """
    Cover = Input(shape=(None, None, 3), name=f'cover_image') # (N, H, W, 3)
    Message = Input(shape=(None, None, D), name=f'message_data') # (N, H, W, D)

    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv')(Cover) # (N, H, W, 3) -> (N, H, W, 32)
    a = BatchNormalization(name='a_normalize')(a)

    b_concatenate = Concatenate(name='b_concatenate')([a, Message]) # (N, H, W, 32) + (N, H, W, D) -> (N, H, W, 32 + D)
    b = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='b_conv')(b_concatenate) # (N, H, W, 32 + D) -> (N, H, W, 32)
    b = BatchNormalization(name='b_normalize')(b)

    c = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='c_conv')(b) # (N, H, W, 32) -> (N, H, W, 32)
    c = BatchNormalization(name='c_normalize')(c)

    d = Conv2D(3, kernel_size=3, padding='same', name='d_conv')(c) # (N, H, W, 32) -> (N, H, W, 3)
    
    Encoder_d = Add(name='add_C_d')([Cover, d]) # (N, H, W, 3) + (N, H, W, 3) -> (N, H, W, 3)
    Encoder_d = Activation('tanh', name='Encoder_activation_tanh')(Encoder_d) # (N, H, W, 3) -> (N, H, W, 3)
    
    return Model(inputs=[Cover, Message], outputs=Encoder_d, name='ResnetSteganoGAN_residual_encoder')

def DenseEncoder(D):
    """
    The DenseEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.
    Input: (N, H, W, 3), (N, H, W, D)
    Output: (N, H, W, 3)
    """
    Cover = Input(shape=(None, None, 3), name=f'cover_image') # (N, H, W, 3)
    Message = Input(shape=(None, None, D), name=f'message_data') # (N, H, W, D)

    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv')(Cover) # (N, H, W, 3) -> (N, H, W, 32)
    a = BatchNormalization(name='a_normalize')(a)

    b_concatenate = Concatenate(name='b_concatenate')([a, Message]) # (N, H, W, 32) + (N, H, W, D) -> (N, H, W, 32 + D)
    b = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='b_conv')(b_concatenate) # (N, H, W, 32 + D) -> (N, H, W, 32)
    b = BatchNormalization(name='b_normalize')(b)

    c_concatenate = Concatenate(name='c_concatenate')([a, b, Message]) # (N, H, W, 32) + (N, H, W, 32) + (N, H, W, D) -> (N, H, W, 64 + D)
    c = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='c_conv')(c_concatenate) # (N, H, W, 64 + D) -> (N, H, W, 32)
    c = BatchNormalization(name='c_normalize')(c)

    d_concatenate = Concatenate(name='d_concatenate')([a, b, c, Message]) # (N, H, W, 32) + (N, H, W, 32) + (N, H, W, 32) + (N, H, W, D) -> (N, H, W, 96 + D)
    d = Conv2D(3, kernel_size=3, padding='same', name='d_conv')(d_concatenate) # (N, H, W, 96 + D) -> (N, H, W, 3)

    Encoder_d = Add(name='add_C_d')([Cover, d]) # (N, H, W, 3) + (N, H, W, 3) -> (N, H, W, 3)
    Encoder_d = Activation('tanh', name='Encoder_activation_tanh')(Encoder_d) # (N, H, W, 3) -> (N, H, W, 3)
    
    return Model(inputs=[Cover, Message], outputs=Encoder_d, name='ResnetSteganoGAN_dense_encoder')

def BasicDecoder(D):
    """
    The BasicDecoder module takes an steganographic image and attempts to decode
    the embedded data tensor.
    Input: (N, H, W, 3)
    Output: (N, H, W, D)
    """
    Cover = Input(shape=(None, None, 3), name=f'cover_image') # (N, H, W, 3)
    
    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv')(Cover) # (N, H, W, 3) -> (N, H, W, 32)
    a = BatchNormalization(name='a_normalize')(a)

    b = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='b_conv')(a) # (N, H, W, 32) -> (N, H, W, 32)
    b = BatchNormalization(name='b_normalize')(b)

    c = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='c_conv')(b) # (N, H, W, 32) -> (N, H, W, 32)
    c = BatchNormalization(name='c_normalize')(c)

    Decoder = Conv2D(D, kernel_size=3, padding='same', name='Decoder_conv')(c) # (N, H, W, 32) -> (N, H, W, D)

    return Model(inputs=Cover, outputs=Decoder, name='ResnetSteganoGAN_basic_decoder')

def DenseDecoder(D):
    """
    The DenseDecoder module takes an steganographic image and attempts to decode
    the embedded data tensor.
    Input: (N, H, W, 3)
    Output: (N, H, W, D)
    """
    Cover = Input(shape=(None, None, 3), name=f'cover_image') # (N, H, W, 3)
    
    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv')(Cover) # (N, H, W, 3) -> (N, H, W, 32)
    a = BatchNormalization(name='a_normalize')(a)

    b = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='b_conv')(a) # (N, H, W, 32) -> (N, H, W, 32)
    b = BatchNormalization(name='b_normalize')(b)

    c_concatenate = Concatenate(name='c_concatenate')([a, b]) # (N, H, W, 32) + (N, H, W, 32) -> (N, H, W, 64)
    c = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='c_conv')(c_concatenate) # (N, H, W, 64) -> (N, H, W, 32)
    c = BatchNormalization(name='c_normalize')(c)

    Decoder_concatenate = Concatenate(name='Decoder_concatenate')([a, b, c]) # (N, H, W, 32) + (N, H, W, 32) + (N, H, W, 32) -> (N, H, W, 96)
    Decoder = Conv2D(D, kernel_size=3, padding='same', name='Decoder_conv')(Decoder_concatenate) # (N, H, W, 96) -> (N, H, W, D)

    return Model(inputs=Cover, outputs=Decoder, name='ResnetSteganoGAN_dense_decoder')

def Discriminator():
    """
    The Discriminator module takes an image and predicts whether it is a cover
    image or a steganographic image (N, 1).
    Input: (N, H, W, 3)
    Output: (N, 1)
    """
    Stego = Input(shape=(None, None, 3), name=f'stego_image') # (N, H, W, 3)

    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv_1')(Stego) # (N, H, W, 3) -> (N, H, W, 32)
    a = MaxPooling2D(pool_size=(2, 2))(a) # (N, H, W, 32) -> (N, H/2, W/2, 32)
    a = Dropout(0.3)(a)

    a = Conv2D(64, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv_2')(a) # (N, H/2, W/2, 32) -> (N, H/2, W/2, 64)
    a = MaxPooling2D(pool_size=(2, 2))(a) # (N, H/2, W/2, 64) -> (N, H/4, W/4, 64)
    a = Dropout(0.3)(a)

    a = Conv2D(128, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv_3')(a) # (N, H/4, W/4, 64) -> (N, H/4, W/4, 128)
    a = MaxPooling2D(pool_size=(2, 2))(a) # (N, H/4, W/4, 128) -> (N, H/8, W/8, 128)
    a = Dropout(0.3)(a)

    global_avarage_score = GlobalAveragePooling2D(name='score_pooling')(a) # (N, H/8, W/8, 128) -> (N, 128) -> (N, 1)
    score = Dense(1, name='score_dense', activation='sigmoid')(global_avarage_score) # (N, 128) -> (N, 1)

    return Model(inputs=Stego, outputs=score, name='ResnetSteganoGAN_discriminator')