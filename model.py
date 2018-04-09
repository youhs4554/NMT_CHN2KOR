import tensorflow as tf
from tensorflow.python.layers.core import Dense
from config import FLAGS
import numpy as np

# model configuration
batch_size = FLAGS.batch_size
num_units = FLAGS.num_units
num_layers = FLAGS.num_layers
dim_emb = FLAGS.dim_emb

def Build_Teaching_Model(src_vocab_size, target_vocab_size):
    # placeholders
    x = tf.placeholder('int32', [batch_size,None])
    y = tf.placeholder('int32', [batch_size,None])
    learning_rate = tf.placeholder('float32')
    keep_prob = tf.placeholder('float32')

    # len_x
    len_x = tf.reduce_sum(tf.sign(x), axis=1)

    # decoding mask
    decoding_mask = tf.cast(tf.sign(y), 'float32')

    src_embeddings = tf.get_variable('src_embeddings', [src_vocab_size, dim_emb])
    target_embeddings = tf.get_variable('target_embeddings', [target_vocab_size, dim_emb])

    # Encoder
    encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units)
    encoder_cell = tf.nn.rnn_cell.DropoutWrapper(encoder_cell, output_keep_prob=keep_prob)
    encoder_cell = tf.nn.rnn_cell.MultiRNNCell([encoder_cell] * num_layers)

    # embedding layer for src language
    encoder_embed = tf.nn.embedding_lookup(src_embeddings, x)

    # encoder projection
    encoder_projection_layer = Dense(num_units, name='encoder_projection')
    encoder_inp = encoder_projection_layer.apply(encoder_embed)

    # RNN Encoder Network
    encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=encoder_inp,
                                                        sequence_length=len_x,
                                                        dtype='float32',
                                                        scope='encoder')

    # Decoder (Dynamic) with attention

    # specify attention mechanism
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=num_units,
                                                               memory=encoder_outputs,
                                                               memory_sequence_length=len_x)

    decoder_cell = []
    for _ in range(num_layers):
        _cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units)
        _cell = tf.nn.rnn_cell.DropoutWrapper(_cell, output_keep_prob=keep_prob)
        decoder_cell.append(_cell)

    decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cell)
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                       attention_layer_size=num_units)  # decoder cell with attention

    y_shifted = tf.concat([tf.fill([batch_size,1], FLAGS.START), y[:,:-1]], 1)
    decoder_inp_embed = tf.nn.embedding_lookup(target_embeddings, y_shifted)

    # sequence_lengths for target language
    sequence_lengths = tf.cast(tf.reduce_sum(decoding_mask, axis=1), 'int32')

    # helper
    helper = tf.contrib.seq2seq.TrainingHelper(decoder_inp_embed, sequence_lengths)

    # decoder output projection layer
    decoder_output_projection_layer = Dense(target_vocab_size, use_bias=False, name='decoder_output_projection')

    # decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, helper, initial_state=decoder_cell.zero_state(dtype='float32', batch_size=batch_size),
        output_layer=decoder_output_projection_layer)

    decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True)

    logits = decoder_outputs.rnn_output

    # loss
    crossent = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=tf.one_hot(y, target_vocab_size, dtype='float32'))

    loss = tf.reduce_sum(crossent * decoding_mask) / tf.reduce_sum(decoding_mask)
    loss_summary = tf.summary.scalar('loss_seq2seq', loss)

    global_step = tf.Variable(0, trainable=False)

    # gradient clip
    # params = tf.trainable_variables()
    #
    # # opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # # gradients = tf.gradients(loss, params)
    # # clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)  # max_gradient_norm
    # # optimizer = opt.apply_gradients(zip(clipped_gradients, params), global_step=global_step)  # decoder


    # Adam
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    return dict(x=x, y=y, len_x=len_x, sequence_lengths=sequence_lengths, learning_rate=learning_rate, keep_prob=keep_prob,                   #### input feeds(i.e. placeholders)
                loss=loss, loss_summary=loss_summary, optimizer=optimizer, global_step=global_step)        #### output feeds

def Build_Decoding_Model(src_vocab_size, target_vocab_size, maxlen_cap=50):
    # placeholders
    x = tf.placeholder('int32', [1,None])

    # len_x
    len_x = tf.reduce_sum(tf.sign(x), axis=1)

    src_embeddings = tf.get_variable('src_embeddings', [src_vocab_size, dim_emb])
    target_embeddings = tf.get_variable('target_embeddings', [target_vocab_size, dim_emb])

    # Encoder
    encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units)
    encoder_cell = tf.nn.rnn_cell.DropoutWrapper(encoder_cell, output_keep_prob=1.0)
    encoder_cell = tf.nn.rnn_cell.MultiRNNCell([encoder_cell] * num_layers)

    # embedding layer for src language
    encoder_embed = tf.nn.embedding_lookup(src_embeddings, x)

    # encoder projection
    encoder_projection_layer = Dense(num_units, name='encoder_projection')
    encoder_inp = encoder_projection_layer.apply(encoder_embed)

    # RNN Encoder Network
    encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=encoder_inp,
                                                        sequence_length=len_x,
                                                        dtype='float32',
                                                        scope='encoder')

    # Decoder (Dynamic) with attention

    # specify attention mechanism
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=num_units,
                                                               memory=encoder_outputs,
                                                               memory_sequence_length=len_x)

    decoder_cell = []
    for _ in range(num_layers):
        _cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units)
        _cell = tf.nn.rnn_cell.DropoutWrapper(_cell, output_keep_prob=1.0)
        decoder_cell.append(_cell)

    decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cell)
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                       attention_layer_size=num_units)  # decoder cell with attention

    # helper
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(target_embeddings,
                                                      tf.fill([1], FLAGS.START),
                                                      FLAGS.END)

    # decoder output projection
    decoder_output_projection_layer = Dense(target_vocab_size, use_bias=False, name='decoder_output_projection')

    # decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, helper, initial_state=decoder_cell.zero_state(dtype='float32', batch_size=1),
        output_layer=decoder_output_projection_layer)

    decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True,
                                                              maximum_iterations=maxlen_cap)

    # generated sentences
    generated = decoder_outputs.sample_id

    return dict(x=x,  #### input feeds(i.e. placeholders)
                generated=generated)  #### output feeds

####
from config import FLAGS
import pickle
from data_utils import Batcher

# load from files
train_set = pickle.load(file('./data/train.pkl', 'rb'))
valid_set = pickle.load(file('./data/valid.pkl', 'rb'))
test_set = pickle.load(file('./data/test.pkl', 'rb'))
word2ix = pickle.load(file('./data/word2ix.pkl', 'rb'))

# prepare ix2word
ix2word = dict(src=None, target=None)
ix2word['src'] = dict(zip(word2ix['src'].values(), word2ix['src'].keys()))
ix2word['target'] = dict(zip(word2ix['target'].values(), word2ix['target'].keys()))

def train():
    train_batcher = Batcher(train_set, batch_size=batch_size)
    valid_batcher = Batcher(valid_set, batch_size=batch_size)

    model = Build_Teaching_Model(len(ix2word['src']),len(ix2word['target']))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    writer_train = tf.summary.FileWriter('./logs/train', graph=sess.graph)
    writer_valid = tf.summary.FileWriter('./logs/valid', graph=sess.graph)

    # train loop
    while True:
        train_batch = train_batcher.next_batch()
        if np.any(np.max(train_batch['src_ixs'], axis=1)==0) or np.any(np.max(train_batch['target_ixs'], axis=1)==0):
            continue

        train_feed = { model['x']:train_batch['src_ixs'], model['y']:train_batch['target_ixs'],
                       model['learning_rate'] : 1e-4,
                       model['keep_prob'] : 0.5 }

        _, train_loss, train_summary = sess.run([model['optimizer'], model['loss'], model['loss_summary']], feed_dict=train_feed)

        step = model['global_step'].eval(sess)

        writer_train.add_summary(train_summary, step)

        print 'step : {0}, train_loss : {1:.4f}'.format(step, train_loss)

        if step % FLAGS.valid_step == 0:
            valid_batch = valid_batcher.next_batch()
            if np.any(np.max(valid_batch['src_ixs'], axis=1) == 0) or np.any(np.max(valid_batch['target_ixs'], axis=1) == 0):
                continue

            valid_feed = {model['x']: valid_batch['src_ixs'], model['y']: valid_batch['target_ixs'],
                          model['keep_prob']: 1.0}

            valid_loss, valid_summary = sess.run([model['loss'], model['loss_summary']], feed_dict=valid_feed)
            writer_valid.add_summary(valid_summary, step)

            print '@@@ step : {0}, valid_loss : {1:.4f}'.format(step, valid_loss)

        if step % FLAGS.save_ckpt_step == 0:
            # save trained model
            saver.save(sess, 'models/step', global_step=step)

def eval():
    valid_batcher = Batcher(valid_set, batch_size=1)

    model = Build_Decoding_Model(len(ix2word['src']),len(ix2word['target']))
    sess = tf.Session()

    #sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    saver.restore(sess, 'models/step-18000')

    while True:
        valid_batch = valid_batcher.next_batch()
        if np.any(np.max(valid_batch['src_ixs'], axis=1) == 0) or np.any(np.max(valid_batch['target_ixs'], axis=1) == 0):
            continue

        valid_feed = {model['x']: valid_batch['src_ixs']}

        generated = sess.run(model['generated'], feed_dict=valid_feed)

        hypo = ' '.join(ix2word['target'][i] for i in generated[0])
        gold = ' '.join(ix2word['target'][i] for i in valid_batch['target_ixs'][0])

        import os
        gold_path = 'result.gold.txt'
        hypo_path = 'result.hypo.txt'

        with file(gold_path, 'w' if not os.path.exists(gold_path) else 'a') as f:
            f.write(gold.encode('utf-8')+'\n')
        with file(hypo_path, 'w' if not os.path.exists(hypo_path) else 'a') as g:
            g.write(hypo.encode('utf-8')+'\n')

train()
#eval()