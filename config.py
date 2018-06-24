import tensorflow as tf

# model config
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.app.flags.DEFINE_integer("num_units", 1024, "number of LSTM units")
tf.app.flags.DEFINE_integer("num_layers", 2, "number of layers of RNN network")
tf.app.flags.DEFINE_integer("dim_emb", 300, "dimension of word embedding")
tf.app.flags.DEFINE_integer("attn_size", 512, "attention layer size")

tf.app.flags.DEFINE_integer("valid_step", 300, "validation step period")
tf.app.flags.DEFINE_integer("save_ckpt_step", 1000, "model's ckpt file saving period")

# fine_tune or not
tf.app.flags.DEFINE_boolean("fine_tune", False, "fine tune or not")

# token config
tf.app.flags.DEFINE_integer("PAD", 0, "token for padding")
tf.app.flags.DEFINE_integer("UNK", 1, 'token for unkown word')
tf.app.flags.DEFINE_integer("START", 2, "token for start of sentence(only target language)")
tf.app.flags.DEFINE_integer("END", 3, "token for end of sentence(only target language)")

# path
tf.app.flags.DEFINE_string("pretrained_model_path", './NMT_save/models', "model save path")
tf.app.flags.DEFINE_string("save_model_path", './NMT_save/models_diary', "model save path")
tf.app.flags.DEFINE_string("log_path", './NMT_save/logs', "model save path")

# which gpu?
tf.app.flags.DEFINE_string("gpu", "0", "Which GPU to use")
# which mode
tf.app.flags.DEFINE_string("mode", None, "Which mode")

FLAGS = tf.app.flags.FLAGS