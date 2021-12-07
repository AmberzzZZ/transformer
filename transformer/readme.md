

## keras MultiHeadAttention layer
    tf2.4.1 keras估计要2.3以上
    https://github.com/keras-team/keras/blob/70d7d07bd186b929d81f7a8ceafff5d78d8bd701/keras/layers/multi_head_attention.py

    given sequence length N, batch size B, key dim d, num_heads m, value_dim dv:

    step1: projects `query`, `key` and `value`, 
    * each is a list of tensors of length `num_attention_heads`
    * each tensor [B, N, d]
    * trainable variables Wq,Wk,Wv(biases)

    step2: compute attention
    * dot(Q,K)
    * scaled by hidden dim d, [B,N,N]
    * softmax to obtain attention probabilities, [B,N,1]
    * dropout，我们的实现中dropout放在MSA layer后面，因为drop的是一整个特征维度，放在哪都行
    * reweight content vec V, [B,N,mdv]

    step3: final dense
    * concat multi-heads along d-axis, [B,N,mdv]
    * linear projection to d, [B,N,dv]



## transformer: attention is all you need
    这里面的attention是transformer attention，established on Q,K,V
    可以用来表征token embedding的不同维度：https://zhuanlan.zhihu.com/p/158952064
    * Q: query，词的查询向量
    * K: key，词的被查向量
    * V: value，词的内容向量

    multi-head self-attention:
    * dq=dk=qv=dmodel/h=64
    * h=8

    encoder:
    N=6, encoder由6个相同的attention block构成
    每个attention block包含：MSA，FF，LN，residual, dropout
        MSA：MultiHeadAttention + add&norm
        FF：dense + relu/gelu + dense
    输入：input embedding + ppositional embedding
    self-attention: single input, x=q=k=v







    
















