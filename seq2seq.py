from keras.models import Model
from keras.layers import Input, LSTM, Dense, Lambda
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np


def seq2seq_teacher(num_encoder_tokens, num_decoder_tokens, latent_dim):
    # encoder
    encoder_inputs = Input(shape=(None, num_encoder_tokens))   # [timestep, emb_dim]
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # decoder: use `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder(decoder_inputs,initial_state=encoder_states)

    # dense prediction
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model


def seq2seq_reinjection(num_encoder_tokens, num_decoder_tokens, latent_dim):
    # encoder
    encoder_inputs = Input(shape=(None, num_encoder_tokens))   # [timestep, emb_dim]
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    states = [state_h, state_c]

    # decoder: use `encoder_states` as initial state.
    decoder_inputs = Input(shape=(1, num_decoder_tokens))    # given a start character is enough
    decoder = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    # loop: run by step, use the current prediction as the next input
    all_outputs = []
    inputs = decoder_inputs   # the initial input
    for _ in range(max_decoder_seq_length):
        outputs, state_h, state_c = decoder(inputs, initial_state=states)
        outputs = decoder_dense(outputs)   # [b,1,emb_dim]
        all_outputs.append(outputs)
        inputs = outputs
        states = [state_h, state_c]

    # Concatenate all predictions
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)   # [b,timestep, emb_dim]

    # model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model


def inference_seq(input_seq, num_encoder_tokens, num_decoder_tokens, latent_dim):

    # encoder model
    encoder_inputs = Input(shape=(None, num_encoder_tokens))   # [timestep, emb_dim]
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.load_weights("seq2seq.h5", by_name=True)

    # decoder model
    decoder_inputs = Input(shape=(1, num_decoder_tokens))   # always input the last prediction
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=[decoder_state_input_h,decoder_state_input_c])
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs, decoder_state_input_h, decoder_state_input_c],
                          [decoder_outputs, state_h, state_c])
    decoder_model.load_weights("seq2seq.h5", by_name=True)

    # encode the inputs into states
    states_value = encoder_model.predict(input_seq)   # used for initial run of the decoder

    # Generate empty target sequence of length 1 as the start character
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the decoder inputs
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]

    return decoded_sentence


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
    # model = seq2seq_teacher(num_encoder_characabulary, num_decoder_characabulary)
    # # model = seq2seq_reinjection(num_encoder_characabulary, num_decoder_characabulary)
    # # model.summary()
    # ckpt = ModelCheckpoint('seq2seq_ep{epoch:02d}.h5', monitor='loss')
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           callbacks=[ckpt],
    #           validation_split=0.2)

    # inference
    x_test = encoder_input_data[-1:]   # [1, timestep, emb_dim]
    pred = inference_seq(x_test, num_encoder_characabulary, num_decoder_characabulary, latent_dim)

    input_sentence = ''
    for vec in x_test[0]:
        index = np.argmax(vec)
        char = reverse_input_char_index[index]
        input_sentence += char

    print(input_sentence)
    print(pred)







