from keras.models import Model
from keras.layers import Input, LSTM, Dense, Lambda, Reshape, Softmax, Concatenate
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def seq2seq_att(N_in, num_encoder_tokens, N_out, num_decoder_tokens, latent_dim, return_att=False):
    # encoder
    encoder_inputs = Input(shape=(N_in, num_encoder_tokens))   # [b,N_in,emb_dim]
    encoder = LSTM(latent_dim, return_sequences=True, return_state=True)
    encoder_outputs, h, _ = encoder(encoder_inputs)    # [b,N_in,encoder_dim]

    # decoder: loop by step
    decoder_inputs = Input(shape=(None, num_decoder_tokens))    # [b,N_out,emb_dim]
    inputs = Lambda(lambda x: tf.split(x, axis=1, num_or_size_splits=N_out))(decoder_inputs)
    state_h = Input(shape=(latent_dim,))
    state_c = Input(shape=(latent_dim,))
    decoder_lstm = LSTM(latent_dim, return_state=True)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = []
    attention_weights = []
    h = state_h    # initial by encoder final state, later is updated by each former-step-outputs
    for i in range(max_decoder_seq_length):
        # compute one-step-attention
        att, att_weights = Lambda(one_step_attention, arguments={'N_in': N_in, 'encoder_dim': latent_dim}) \
                           ([encoder_outputs, inputs[i]])
        attention_weights.append(att_weights)    # [b,N_in]
        out, ht, _ = decoder_lstm(inputs[i], initial_state=[h, att])
        h = ht
        out = decoder_dense(out)   # [b,1,emb_dim]
        decoder_outputs.append(out)
    decoder_outputs = Lambda(lambda x: tf.stack(x, axis=1))(decoder_outputs)    # [b,N_out,decoder_dim]

    # model
    if return_att:
        model = Model([encoder_inputs, decoder_inputs, state_h, state_c], [decoder_outputs]+attention_weights)
    else:
        model = Model([encoder_inputs, decoder_inputs, state_h, state_c], decoder_outputs)

    return model


def one_step_attention(args, N_in, encoder_dim):
    h, s = args
    # h: [b, N_in, encoder_dim]
    # s: [b, 1, decoder_dim]
    s = tf.tile(s, [1,N_in,1])
    a = tf.concat([h,s], axis=-1)    # [b, N_in, reltate_dim]
    # compute trainable relationship along axis-2
    alpha = Dense(32, activation='tanh')(a)    # [b, N_in, 32]
    alpha = Dense(1, activation='relu')(a)    # [b, N_in, 1]
    # compute weights along axis-1
    alpha = Softmax(axis=1)(alpha)    # [b, N_in], attention weights!!
    # reweight context vec
    alpha1 = tf.tile(alpha, [1,1,encoder_dim])
    return [K.sum(h*alpha1, axis=1), alpha]    # [b,encoder_dim]


def inference_seq(model, input_seq, N_in, N_out):

    target_seq = np.zeros((1, max_decoder_seq_length, num_decoder_characabulary))
    target_seq[0, 0, target_token_index['\t']] = 1.
    shadow_h = np.zeros((1,latent_dim))
    shadow_c = np.zeros((1,latent_dim))
    preds = model.predict([x_test, target_seq, shadow_h, shadow_c])
    pred = preds[0][0]   # [N_out, out_dim]
    atts = preds[1:]    # N_out elements of [b,N_in]
    decoded_sentence = ''
    attention_map = np.zeros((N_out, N_in))
    for i in range(max_decoder_seq_length):
        # inference out_seq
        sampled_token_index = np.argmax(pred[i])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        # Exit condition
        if sampled_char == '\n':
            break
        # compute att map
        for j in range(N_in):
            attention_map[i][j] = atts[i][0][j]

    input_sentence = ''
    for vec in x_test[0]:
        index = np.argmax(vec)
        char = reverse_input_char_index[index]
        input_sentence += char
    print("input sentence: ", input_sentence)
    print("predicted translated sentence: ", decoded_sentence)

    # vis attention
    source_list = input_sentence.split()
    target_list = decoded_sentence.split()
    f, ax = plt.subplots(figsize=(N_out,N_in))
    sns.heatmap(attention_map, xticklabels=source_list, yticklabels=target_list, cmap="YlGnBu")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    plt.show()


if __name__ == '__main__':

    # data process
    batch_size = 256
    epochs = 100
    latent_dim = 256     # lstm units dim
    num_samples = 10000
    mode = 'teacher_forcing'  # 'reinjection'
    data_path = '/Users/amber/dataset/fra-eng/fra.txt'

    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    for line in lines[: min(num_samples, len(lines) - 1)]:
        target_text, input_text = line.split('\t')[:2]
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_characabulary = len(input_characters)
    num_decoder_characabulary = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_characabulary)
    print('Number of unique output tokens:', num_decoder_characabulary)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    # characabulary
    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])   # for coding the one-hot label
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())  # for parsing the inference results
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

    # one-hot training data
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_characabulary), dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_characabulary), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_characabulary), dtype='float32')
    # batch idx i, charac idx t, one-hot idx dict[char]
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        encoder_input_data[i, t+1:, input_token_index[' ']] = 1.
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                decoder_target_data[i, t-1, target_token_index[char]] = 1.
        decoder_input_data[i, t+1:, target_token_index[' ']] = 1.
        decoder_target_data[i, t:, target_token_index[' ']] = 1.
    if mode != 'teacher_forcing':
        decoder_input_data = np.zeros((len(input_texts), 1, num_decoder_characabulary), dtype='float32')
        decoder_input_data[:, 0, target_token_index['\t']] = 1.

    # # training
    # model = seq2seq_att(max_encoder_seq_length, num_encoder_characabulary,
    #                     max_decoder_seq_length, num_decoder_characabulary, latent_dim)
    # model.load_weights("seq2seq_att_ep06.h5")
    # ckpt = ModelCheckpoint('seq2seq_att_ep{epoch:02d}.h5', monitor='loss')
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # shadow_h = np.zeros((num_samples,latent_dim))
    # shadow_c = np.zeros((num_samples,latent_dim))
    # model.fit([encoder_input_data, decoder_input_data, shadow_h, shadow_c], decoder_target_data,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           callbacks=[ckpt],
    #           validation_split=0.2)

    # inference
    model = seq2seq_att(max_encoder_seq_length, num_encoder_characabulary,
                        max_decoder_seq_length, num_decoder_characabulary, latent_dim,
                        return_att=True)
    model.load_weights("seq2seq_att_ep10.h5")
    x_test = encoder_input_data[-1:]   # [1, timestep, emb_dim]
    # target_seq = np.zeros((1, max_decoder_seq_length, num_decoder_characabulary))
    # target_seq[0, 0, target_token_index['\t']] = 1.
    # shadow_h = np.zeros((1,latent_dim))
    # shadow_c = np.zeros((1,latent_dim))
    # pred = model.predict([x_test, target_seq, shadow_h, shadow_c])[0]
    # print(pred)
    # decoded_sentence = ''
    # for i in range(max_decoder_seq_length):
    #     sampled_token_index = np.argmax(pred[i])
    #     sampled_char = reverse_target_char_index[sampled_token_index]
    #     decoded_sentence += sampled_char
    # print(decoded_sentence)

    # input_sentence = ''
    # for vec in x_test[0]:
    #     index = np.argmax(vec)
    #     char = reverse_input_char_index[index]
    #     input_sentence += char
    # print(input_sentence)

    # vis att
    inference_seq(model, x_test, max_encoder_seq_length, max_decoder_seq_length)








