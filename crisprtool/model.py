from keras import Model
from keras import regularizers
from keras.layers import Conv2D, BatchNormalization, ReLU, Input, Flatten, Convolution1D
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers import AveragePooling1D, Bidirectional, LSTM, GlobalAveragePooling1D
from keras.layers import LayerNormalization, Conv1D, MultiHeadAttention, Layer, SimpleRNN
import numpy as np
import tensorflow as tf



def DeepCRISPR(input_shape):
    inputs_sg = Input(shape=input_shape)
    x = inputs_sg

    # Encoder
    x = Conv2D(8, kernel_size=[1, 3], padding='valid', name='e_1')(x)
    x = BatchNormalization(momentum=0, center=False, scale=False, name='ebn_1u')(x)
    x = ReLU()(x)

    x = Conv2D(32, kernel_size=[1, 3], strides=1, padding='valid', name='e_2')(x)
    x = BatchNormalization(momentum=0, center=False, scale=False, name='ebn_2u')(x)
    x = ReLU()(x)

    x = Conv2D(64, kernel_size=[1, 3], padding='valid', name='e_3')(x)
    x = BatchNormalization(momentum=0, center=False, scale=False, name='ebn_3u')(x)
    x = ReLU()(x)

    x = Conv2D(64, kernel_size=[1, 3], strides=1, padding='valid', name='e_4')(x)
    x = BatchNormalization(momentum=0, center=False, scale=False, name='ebn_4u')(x)
    x = ReLU()(x)

    x = Conv2D(256, kernel_size=[1, 3], padding='valid', name='e_5')(x)
    x = BatchNormalization(momentum=0, center=False, scale=False, name='ebn_5u')(x)
    x = ReLU()(x)

    # regressor
    x = Conv2D(512, kernel_size=[1, 3], strides=2, padding='valid', name='e_6')(x)
    x = BatchNormalization(momentum=0.99, center=False, scale=False, name='ebn_6l')(x)
    x = ReLU()(x)

    x = Conv2D(512, kernel_size=[1, 3], padding='valid', name='e_7')(x)
    x = BatchNormalization(momentum=0.99, center=False, scale=False, name='ebn_7l')(x)
    x = ReLU()(x)

    x = Conv2D(1024, kernel_size=[1, 3], padding='valid', name='e_8')(x)
    x = BatchNormalization(momentum=0.99, center=False, scale=False, name='ebn_8l')(x)
    x = ReLU()(x)

    # Add GlobalAveragePooling2D before Dense layer
    x = GlobalAveragePooling2D()(x)

    # Replace Conv2D layer with Dense layer
    x = Dense(1, activation = "linear", name='e_9')(x)

    model = Model(inputs_sg, x)
    return model


def Cas9_BiLSTM(input_shape):
    input = Input(shape=input_shape)

    conv1 = Conv1D(128, 3, activation="relu")(input)
    pool1 = AveragePooling1D(2)(conv1)
    drop1 = Dropout(0.4)(pool1)

    conv2 = Conv1D(128, 3, activation="relu")(drop1)
    pool2 = AveragePooling1D(2)(conv2)
    drop2 = Dropout(0.4)(pool2)

    lstm1 = Bidirectional(LSTM(32,
                               dropout=0.4,
                               activation='tanh',
                               return_sequences=True,
                               kernel_regularizer=regularizers.l2(1e-4)))(drop2)
    avgpool = GlobalAveragePooling1D()(lstm1)

    dense1 = Dense(128,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(avgpool)
    drop3 = Dropout(0.4)(dense1)

    dense2 = Dense(64,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(drop3)
    drop4 = Dropout(0.4)(dense2)

    dense3 = Dense(128,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(drop4)
    drop5 = Dropout(0.4)(dense3)

    output = Dense(1, activation="linear")(drop5)

    model = Model(inputs=[input], outputs=[output])
    return model


def Cas9_SimpleRNN(input_shape):
    input = Input(shape=input_shape)

    conv1 = Conv1D(64, 3, activation="relu")(input)
    pool1 = AveragePooling1D(2)(conv1)
    drop1 = Dropout(0.1)(pool1)

    conv2 = Conv1D(64, 3, activation="relu")(drop1)
    pool2 = AveragePooling1D(2)(conv2)
    drop2 = Dropout(0.1)(pool2)

    srnn1 = SimpleRNN(64,
                      dropout=0.3,
                      activation="tanh",
                      return_sequences=True,
                      kernel_regularizer=regularizers.l2(0.01))(drop2)
    srnn2 = SimpleRNN(128,
                      dropout=0.3,
                      activation="tanh",
                      return_sequences=True,
                      kernel_regularizer=regularizers.l2(0.01))(srnn1)
    avgpool = GlobalAveragePooling1D()(srnn2)

    dense1 = Dense(512,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(avgpool)
    drop3 = Dropout(0.1)(dense1)

    dense2 = Dense(128,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(drop3)
    drop4 = Dropout(0.1)(dense2)

    dense3 = Dense(256,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(drop4)
    drop5 = Dropout(0.1)(dense3)

    output = Dense(1, activation='linear')(drop5)
    model = Model(inputs=[input], outputs=[output])

    return model




class PositionalEncoding(Layer):
    def __init__(self, sequence_len=None, embedding_dim=None,**kwargs):
        super(PositionalEncoding, self).__init__()
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim

    def call(self, x):

        position_embedding = np.array([
            [pos / np.power(10000, 2. * i / self.embedding_dim) for i in range(self.embedding_dim)]
            for pos in range(self.sequence_len)])

        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])  # dim 2i
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])  # dim 2i+1
        position_embedding = tf.cast(position_embedding, dtype=tf.float32)

        return position_embedding+x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sequence_len' : self.sequence_len,
            'embedding_dim' : self.embedding_dim,
        })
        return config


def Cas9_MultiHeadAttention(input_shape):
    input = Input(shape=input_shape)

    conv1 = Conv1D(256, 3, activation="relu")(input)
    pool1 = AveragePooling1D(2)(conv1)
    drop1 = Dropout(0.4)(pool1)

    conv2 = Conv1D(256, 3, activation="relu")(drop1)
    pool2 = AveragePooling1D(2)(conv2)
    drop2 = Dropout(0.4)(pool2)

    lstm = Bidirectional(LSTM(128,
                               dropout=0.5,
                               activation='tanh',
                               return_sequences=True,
                               kernel_regularizer=regularizers.l2(0.01)))(drop2)

    pos_embedding = PositionalEncoding(sequence_len=int(((23-3+1)/2-3+1)/2), embedding_dim=2*128)(lstm)
    atten = MultiHeadAttention(num_heads=2,
                               key_dim=64,
                               dropout=0.2,
                               kernel_regularizer=regularizers.l2(0.01))(pos_embedding, pos_embedding)

    flat = Flatten()(atten)

    dense1 = Dense(512,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(flat)
    drop3 = Dropout(0.1)(dense1)

    dense2 = Dense(128,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(drop3)
    drop4 = Dropout(0.1)(dense2)

    dense3 = Dense(256,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(drop4)
    drop5 = Dropout(0.1)(dense3)

    output = Dense(1, activation="linear")(drop5)

    model = Model(inputs=[input], outputs=[output])
    return model



class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate):
        # embed_dim: Embedding size for each token
        # num_heads: Number of attention heads
        # ff_dim: Hidden layer size in feed forward network inside transformer

        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"),
             Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-3)
        self.layernorm2 = LayerNormalization(epsilon=1e-3)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=None, **kwargs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def Cas9_Transformer(input_shape):
    input = Input(shape=input_shape)
    conv1 = Conv1D(512, 3, activation="relu")(input)
    pool1 = AveragePooling1D(2)(conv1)
    drop1 = Dropout(0.1)(pool1)

    conv2 = Conv1D(512, 3, activation="relu")(drop1)
    pool2 = AveragePooling1D(2)(conv2)
    drop2 = Dropout(0.1)(pool2)

    lstm1 = Bidirectional(LSTM(16,
                               dropout=0.5,
                               activation='tanh',
                               return_sequences=True,
                               kernel_regularizer=regularizers.l2(0.01)))(drop2)
    lstm2 = Bidirectional(LSTM(16,
                               dropout=0.5,
                               activation='tanh',
                               return_sequences=True,
                               kernel_regularizer=regularizers.l2(0.01)))(lstm1)

    pos_embedding = PositionalEncoding(sequence_len=int(((23-3+1)/2-3+1)/2), embedding_dim=2*16)(lstm2)
    trans = TransformerBlock(embed_dim=2*16, num_heads=6, ff_dim=128, dropout_rate=0.1)(pos_embedding)
    avgpool = GlobalAveragePooling1D()(trans)

    dense1 = Dense(512,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(avgpool)
    drop3 = Dropout(0.1)(dense1)

    dense2 = Dense(64,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(drop3)
    drop4 = Dropout(0.1)(dense2)

    dense3 = Dense(16,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(drop4)
    drop5 = Dropout(0.1)(dense3)

    output = Dense(1, activation="linear")(drop5)

    model = Model(inputs=[input], outputs=[output])
    return model



def Seq_deepCpf1(input_shape):
    Seq_deepCpf1_Input_SEQ = Input(shape=input_shape)

    Seq_deepCpf1_C1 = Convolution1D(80, 5, activation='relu')(Seq_deepCpf1_Input_SEQ)
    Seq_deepCpf1_P1 = AveragePooling1D(2)(Seq_deepCpf1_C1)
    Seq_deepCpf1_F = Flatten()(Seq_deepCpf1_P1)
    Seq_deepCpf1_DO1 = Dropout(0.3)(Seq_deepCpf1_F)
    Seq_deepCpf1_D1 = Dense(80, activation='relu')(Seq_deepCpf1_DO1)
    Seq_deepCpf1_DO2 = Dropout(0.3)(Seq_deepCpf1_D1)
    Seq_deepCpf1_D2 = Dense(40, activation='relu')(Seq_deepCpf1_DO2)
    Seq_deepCpf1_DO3 = Dropout(0.3)(Seq_deepCpf1_D2)
    Seq_deepCpf1_D3 = Dense(40, activation='relu')(Seq_deepCpf1_DO3)
    Seq_deepCpf1_DO4 = Dropout(0.3)(Seq_deepCpf1_D3)

    Seq_deepCpf1_Output = Dense(1, activation='linear')(Seq_deepCpf1_DO4)
    Seq_deepCpf1 = Model(inputs=[Seq_deepCpf1_Input_SEQ], outputs=[Seq_deepCpf1_Output])
    return Seq_deepCpf1



def Cas12_SimpleRNN(input_shape):
    dropout_rate = 0.2
    input = Input(shape=input_shape)

    conv1 = Conv1D(128, 5, activation="relu")(input)
    pool1 = AveragePooling1D(2)(conv1)
    drop1 = Dropout(dropout_rate)(pool1)

    conv2 = Conv1D(128, 5, activation="relu")(drop1)
    pool2 = AveragePooling1D(2)(conv2)
    drop2 = Dropout(dropout_rate)(pool2)

    srnn1 = SimpleRNN(32,
                      dropout=dropout_rate,
                      activation="tanh",
                      return_sequences=True,
                      kernel_regularizer=regularizers.l2(0.01))(drop2)
    srnn2 = SimpleRNN(32,
                      dropout=dropout_rate,
                      activation="tanh",
                      return_sequences=True,
                      kernel_regularizer=regularizers.l2(0.01))(srnn1)
    avgpool = GlobalAveragePooling1D()(srnn2)

    dense1 = Dense(512,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(avgpool)
    drop3 = Dropout(dropout_rate)(dense1)

    dense2 = Dense(512,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(drop3)
    drop4 = Dropout(dropout_rate)(dense2)

    dense3 = Dense(512,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(drop4)
    drop5 = Dropout(dropout_rate)(dense3)

    output = Dense(1, activation='linear')(drop5)
    model = Model(inputs=[input], outputs=[output])

    return model



def Cas12_BiLSTM(input_shape):
    input = Input(shape=input_shape)

    conv1 = Conv1D(128, 5, activation="relu")(input)
    pool1 = AveragePooling1D(2)(conv1)
    drop1 = Dropout(0.1)(pool1)

    conv2 = Conv1D(128, 5, activation="relu")(drop1)
    pool2 = AveragePooling1D(2)(conv2)
    drop2 = Dropout(0.1)(pool2)

    lstm1 = Bidirectional(LSTM(128,
                               dropout=0.1,
                               activation='tanh',
                               return_sequences=True,
                               kernel_regularizer=regularizers.l2(1e-4)))(drop2)
    avgpool = GlobalAveragePooling1D()(lstm1)

    dense1 = Dense(128,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(avgpool)
    drop3 = Dropout(0.1)(dense1)

    dense2 = Dense(32,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(drop3)
    drop4 = Dropout(0.1)(dense2)

    dense3 = Dense(32,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(drop4)
    drop5 = Dropout(0.1)(dense3)

    output = Dense(1, activation="linear")(drop5)

    model = Model(inputs=[input], outputs=[output])
    return model



def Cas12_MultiHeadAttention(input_shape):
    input = Input(shape=input_shape)

    conv1 = Conv1D(512, 5, activation="relu")(input)
    pool1 = AveragePooling1D(2)(conv1)
    drop1 = Dropout(0.4)(pool1)

    conv2 = Conv1D(512, 5, activation="relu")(drop1)
    pool2 = AveragePooling1D(2)(conv2)
    drop2 = Dropout(0.4)(pool2)

    lstm = Bidirectional(LSTM(16,
                               dropout=0.5,
                               activation='tanh',
                               return_sequences=True,
                               kernel_regularizer=regularizers.l2(0.01)))(drop2)

    pos_embedding = PositionalEncoding(sequence_len=int(((34-5+1)/2-5+1)/2), embedding_dim=2*16)(lstm)
    atten = MultiHeadAttention(num_heads=2,
                               key_dim=32,
                               dropout=0.5,
                               kernel_regularizer=regularizers.l2(0.01))(pos_embedding, pos_embedding)

    flat = Flatten()(atten)

    dense1 = Dense(256,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(flat)
    drop3 = Dropout(0.2)(dense1)

    dense2 = Dense(128,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(drop3)
    drop4 = Dropout(0.2)(dense2)

    dense3 = Dense(512,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(drop4)
    drop5 = Dropout(0.2)(dense3)

    output = Dense(1, activation="linear")(drop5)

    model = Model(inputs=[input], outputs=[output])
    return model



def Cas12_Transformer(input_shape):
    input = Input(shape=input_shape)
    conv1 = Conv1D(512, 5, activation="relu")(input)
    pool1 = AveragePooling1D(2)(conv1)
    drop1 = Dropout(0.4)(pool1)

    conv2 = Conv1D(512, 5, activation="relu")(drop1)
    pool2 = AveragePooling1D(2)(conv2)
    drop2 = Dropout(0.4)(pool2)

    lstm1 = Bidirectional(LSTM(32,
                               dropout=0.2,
                               activation='tanh',
                               return_sequences=True,
                               kernel_regularizer=regularizers.l2(0.01)))(drop2)
    lstm2 = Bidirectional(LSTM(64,
                               dropout=0.2,
                               activation='tanh',
                               return_sequences=True,
                               kernel_regularizer=regularizers.l2(0.01)))(lstm1)

    pos_embedding = PositionalEncoding(sequence_len=int(((34-5+1)/2-5+1)/2), embedding_dim=2*64)(lstm2)
    trans = TransformerBlock(embed_dim=2*64, num_heads=2, ff_dim=256, dropout_rate=0.3)(pos_embedding)
    avgpool = GlobalAveragePooling1D()(trans)

    dense1 = Dense(512,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(avgpool)
    drop3 = Dropout(0.1)(dense1)

    dense2 = Dense(256,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(drop3)
    drop4 = Dropout(0.1)(dense2)

    dense3 = Dense(16,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu")(drop4)
    drop5 = Dropout(0.1)(dense3)

    output = Dense(1, activation="linear")(drop5)

    model = Model(inputs=[input], outputs=[output])
    return model






