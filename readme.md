## seq2seq model
    
    example: character-level english to french

    ### 数据
    数据集格式: english sentence \t french sentence \t ...
    target: english
    input: french
    起始符&终止符: '\t' + decoder_input_sentence + '\n'
    characabulary: 有字母，数字，空格，标点符号，decoder_input_sentence还有起始符&终止符，decoder_target_sentence比它早一步，没有起始符，有终止符
    one-hot: sentence范围内one-hot for each charac, 其余的填充用空格的one-hot

    ### data dim
    encoder_input_data: 
        [batch_size, max_sentence_length, num_eng_characters]
        each element is a one-hot vec (standing for a specific charac)

    decoder_input_data: 
        [batch_size, max_sentence_length, num_eng_characters]
        each element is a one-hot vec

    decoder_target_data: 
        [batch_size, max_sentence_length, num_eng_characters]
        offset the decoder_input_data by one step: 
            decoder_target比decoder_input早一步，decoder这一步的预测/gt作为下一步的input
            decoder_target_data[:,t,:] = decoder_input_data[:,t+1,:]

    ### model
        model input: 
            encoder input: time-distributed input sequence
            decoder input: target sequence starting with \t
            期望输入数据尺寸: (batch_size, timesteps, data_dim)
        model output: predict sequence, (batch_size, timesteps, data_dim)
        model target: target sequence, (batch_size, timesteps, data_dim)

        * 如果使用函数式API的Model类模型，我们会定义input layer的shape=(timestep, emb_dim)
        * 如果使用Sequential模型，我们要在第一个LSTM层里显示定义input_shape=(timesteps, data_dim)
        * batch_size is set for axis-0 by default for both the methods above
        * 但是在stateful的LSTM layer里面，batch_size要被显示声明：
            model.add(LSTM(32, return_sequences=True, stateful=True, batch_input_shape=(batch_size, timesteps, data_dim)))

    ### teacher-forcing or reinjection
    decoder的输出是预测sequence（gt是与之对应的target sequence），输入是比预测序列one-step-lag-behind的输出序列————你只能拿到并复用已经预测出的东西
    如果用teacher-forcing，decoder的输入是target sequence，
    如果用reinjection，decoder的输入是predicted sequence
    【QUESTION】为啥teacher-forcing更常使用？

    what if:
    * GRU
    * word-level
