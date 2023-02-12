# 0x11. Attention #

<img src="https://github.com/RayBar72/holbertonschool-machine_learning/blob/master/image.png" width="1000" height="450">

## Learning Objectives ##

- What is the attention mechanism?
- How to apply attention to RNNs
- What is a transformer?
- How to create an encoder-decoder transformer model
- What is GPT?
- What is BERT?
- What is self-supervised learning?
- How to use BERT for specific NLP tasks
- What is SQuAD? GLUE?

## Content Table ##

| Task | Description | File |
| ----------- | ----------- | ----------- |
| 0. RNN Encoder | Resources | 0-rnn_encoder.py |
| 1. Self Attention | Create a class SelfAttention that inherits from tensorflow.keras.layers.Layer to calculate the attention for machine translation based on this paper | 1-self_attention.py |
| 2. RNN Decoder | Create a class RNNDecoder that inherits from tensorflow.keras.layers.Layer to decode for machine translation | 2-rnn_decoder.py |
| 3. Positional Encoding | Write the function def positional_encoding(max_seq_len, dm): that calculates the positional encoding for a transformer | 4-positional_encoding.py |
| 4. Scaled Dot Product Attention | Write the function def sdp_attention(Q, K, V, mask=None) that calculates the scaled dot product attention | 5-sdp_attention.py |
| 5. Multi Head Attention | Why multi-head self attention works: math, intuitions and 10+1 hidden insights | 6-multihead_attention.py |
| 6. Transformer Encoder Block | Create a class EncoderBlock that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer | 7-transformer_encoder_block.py |
| 7. Transformer Decoder Block | Create a class DecoderBlock that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer | 8-transformer_decoder_block.py |
| 8. Transformer Encoder | Create a class Encoder that inherits from tensorflow.keras.layers.Layer to create the encoder for a transformer | 9-transformer_encoder.py |
| 9. Transformer Decoder | Create a class Decoder that inherits from tensorflow.keras.layers.Layer to create the decoder for a transformer | 10-transformer_decoder.py |
| 10. Transformer Network | Create a class Transformer that inherits from tensorflow.keras.Model to create a transformer network | 11-transformer.py |

## Authors: ##

**Solution by:** Raymundo Barrera Flores. [rbarreraf72@gmail.com](rbarreraf72@gmail.com)
[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>](https://www.linkedin.com/in/raymundo-barrera-flores-a13022222/)


**Project Required by**: HolbertonSchool
