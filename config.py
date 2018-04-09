import tensorflow as tf

# model config
tf.app.flags.DEFINE_integer("batch_size", 16, "batch size")
tf.app.flags.DEFINE_integer("num_units", 1024, "number of LSTM units")
tf.app.flags.DEFINE_integer("num_layers", 2, "number of layers of RNN network")
tf.app.flags.DEFINE_integer("dim_emb", 300, "dimension of word embedding")
tf.app.flags.DEFINE_integer("attn_size", 512, "attention layer size")

tf.app.flags.DEFINE_integer("valid_step", 100, "validation step period")
tf.app.flags.DEFINE_integer("save_ckpt_step", 1000, "model's ckpt file saving period")

# token config
tf.app.flags.DEFINE_integer("PAD", 0, "token for padding")
tf.app.flags.DEFINE_integer("UNK", 1, 'token for unkown word')
tf.app.flags.DEFINE_integer("START", 2, "token for start of sentence(only target language)")
tf.app.flags.DEFINE_integer("END", 3, "token for end of sentence(only target language)")

FLAGS = tf.app.flags.FLAGS
