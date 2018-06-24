import tensorflow as tf
from tensorflow.python.layers.core import Dense
from config import FLAGS
import os, sys
import numpy as np

# model configuration
batch_size = FLAGS.batch_size
num_units = FLAGS.num_units
num_layers = FLAGS.num_layers
dim_emb = FLAGS.dim_emb

####
from config import FLAGS
import json
from data_utils import Batcher

# load from files
train_set = json.load(file('./data/train.json', 'rb'))
valid_set = json.load(file('./data/valid.json', 'rb'))
test_set = json.load(file('./data/test.json', 'rb'))

word2ix = json.load(file('./data/word2ix.json', 'rb'))

# prepare ix2word
ix2word = dict(src=None, target=None)
ix2word['src'] = dict(zip(word2ix['src'].values(), word2ix['src'].keys()))
ix2word['target'] = dict(zip(word2ix['target'].values(), word2ix['target'].keys()))

os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

def RNNCellWrapper(num_units, num_layers, keep_prob, attention=False, attention_mechanism=None, attention_layer_size=None):
    cells = []
    for _ in range(num_layers):
        cell = tf.nn.rnn_cell.GRUCell(num_units=num_units)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        cells.append(cell)

    cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    if attention:
        # decoder cell with attention
        cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=attention_layer_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    return cell


def Build_NMT_Model(x,y,decoding_mask, keep_prob,
                    src_embeddings, target_embeddings, is_train, maxlen=100):
    # len_x
    len_x = tf.reduce_sum(tf.sign(x), axis=1)

    # Encoder (bidirectional)
    encoder_cell_fw = RNNCellWrapper(num_units=num_units,
                                     num_layers=num_layers,
                                     keep_prob= keep_prob)

    encoder_cell_bw = RNNCellWrapper(num_units=num_units,
                                     num_layers=num_layers,
                                     keep_prob= keep_prob)

    # embedding layer for src language
    encoder_embed = tf.nn.embedding_lookup(src_embeddings, x)

    # RNN Encoder Network
    encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw,
                                                                     cell_bw=encoder_cell_bw,
                                                                     inputs=encoder_embed,
                                                                     dtype='float32',
                                                                     scope='encoder')

    encoder_outputs_concat = tf.concat(encoder_outputs, axis=-1)

    # specify attention mechanism
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=num_units,
                                                               memory=encoder_outputs_concat,
                                                               memory_sequence_length=len_x)

    decoder_cell = RNNCellWrapper(num_units=num_units,
                                  num_layers=num_layers,
                                  keep_prob=keep_prob,
                                  attention=True, attention_mechanism=attention_mechanism, attention_layer_size=num_units)

    if is_train:
        y_shifted = tf.concat([tf.fill([batch_size, 1], FLAGS.START), y[:, :-1]], 1)
        decoder_inp_embed = tf.nn.embedding_lookup(target_embeddings, y_shifted)

        # sequence_lengths for target language
        sequence_lengths = tf.cast(tf.reduce_sum(decoding_mask, axis=1), 'int32')

        # train helper (teaching for training mode)
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_inp_embed, sequence_lengths)
    else:
        # greedy embedding helper (greedy decoding mode)
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(target_embeddings,
                                                          tf.fill([batch_size], FLAGS.START),
                                                          FLAGS.END)

    # decoder projection
    decoder_projection_layer = Dense(target_embeddings.get_shape().as_list()[0], use_bias=False, name='decoder_projection')

    # decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, helper, initial_state=
        decoder_cell.zero_state(dtype='float32', batch_size=batch_size),
        output_layer=decoder_projection_layer)

    decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                              maximum_iterations=None if is_train else maxlen)

    return decoder_outputs


def train():
    train_batcher = Batcher(train_set, batch_size=batch_size)
    valid_batcher = Batcher(valid_set, batch_size=batch_size)

    src_vocab_size = len(ix2word['src'])
    target_vocab_size = len(ix2word['target'])

    # placeholders
    x = tf.placeholder('int32', [batch_size,None])
    y = tf.placeholder('int32', [batch_size,None])
    learning_rate = tf.placeholder('float32')
    keep_prob = tf.placeholder('float32')

    # decoding mask
    decoding_mask = tf.cast(tf.sign(y), 'float32')

    train_src = True
    train_target = False if FLAGS.fine_tune else True

    src_embeddings = tf.get_variable('src_embeddings', [src_vocab_size, dim_emb], trainable=train_src)
    target_embeddings = tf.get_variable('target_embeddings', [target_vocab_size, dim_emb], trainable=train_target)

    decoder_outputs = Build_NMT_Model(x=x,y=y,decoding_mask=decoding_mask, keep_prob=keep_prob,
                                      src_embeddings=src_embeddings, target_embeddings=target_embeddings, is_train=True)

    # logits
    logits = decoder_outputs.rnn_output

    # loss
    crossent = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=tf.one_hot(y, target_vocab_size, dtype='float32'))

    loss = tf.reduce_sum(crossent * decoding_mask) / tf.reduce_sum(decoding_mask)
    loss_summary = tf.summary.scalar('loss_seq2seq', loss)

    global_step = tf.Variable(0, trainable=False)

    # Adam
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    var_list = tf.trainable_variables()
    if FLAGS.fine_tune:
        var_list.append(target_embeddings)

    # savers
    saver = tf.train.Saver(var_list=var_list)

    pretrained_model_path = FLAGS.pretrained_model_path
    save_model_path = FLAGS.save_model_path

    if not FLAGS.fine_tune:
         save_model_path = pretrained_model_path

    if FLAGS.fine_tune:
        ckpt = tf.train.latest_checkpoint(pretrained_model_path)
        print 'restore pre-trained params from {} ...'.format(os.path.splitext(ckpt)[0])
        saver.restore(sess, ckpt)

    trace_ckpt = tf.train.latest_checkpoint(save_model_path)
    if trace_ckpt:
        print 'restore all params from {} ...'.format(os.path.splitext(trace_ckpt)[0])
        saver.restore(sess, trace_ckpt)

    writer_train = tf.summary.FileWriter(os.path.join(FLAGS.log_path,'train'), graph=sess.graph)
    writer_valid = tf.summary.FileWriter(os.path.join(FLAGS.log_path,'valid'), graph=sess.graph)

    # train loop
    last_100_losses = []
    while True:
        train_batch = train_batcher.next_batch()
        if np.any(np.max(train_batch['src_ixs'], axis=1)==0) or np.any(np.max(train_batch['target_ixs'], axis=1)==0): continue
        train_feed = { x:train_batch['src_ixs'], y:train_batch['target_ixs'],
                       learning_rate : 1e-4, keep_prob : 0.5 }

        _, train_loss, train_summary = sess.run([optimizer, loss, loss_summary], feed_dict=train_feed)


        step = global_step.eval(sess)

        summerize = step % 100 == 0

        writer_train.add_summary(train_summary, step)

        last_100_losses.append(train_loss)

        if summerize:
            print('step : {0}, avg_train_loss : {1:.4f}'.format(step, np.mean(last_100_losses)))
            last_100_losses = []

        if step % FLAGS.valid_step == 0:
            valid_batch = valid_batcher.next_batch()
            if np.any(np.max(valid_batch['src_ixs'], axis=1) == 0) or np.any(
                np.max(valid_batch['target_ixs'], axis=1) == 0): continue
            valid_feed = { x: valid_batch['src_ixs'], y: valid_batch['target_ixs'] , keep_prob : 1.0 }

            valid_loss, valid_summary = sess.run([loss, loss_summary], feed_dict=valid_feed)
            writer_valid.add_summary(valid_summary, step)

            print('@@@ step : {0}, valid_loss : {1:.4f}'.format(step, valid_loss))

        if step % FLAGS.save_ckpt_step == 0:
            # save trained model
            saver.save(sess, os.path.join(save_model_path,'step'), global_step=step)



def eval():
    batcher = Batcher(test_set, batch_size=batch_size)

    src_vocab_size = len(ix2word['src'])
    target_vocab_size = len(ix2word['target'])

    x = tf.placeholder('int32', [batch_size,None])
    keep_prob = tf.placeholder('float32')

    src_embeddings = tf.get_variable('src_embeddings', [src_vocab_size, dim_emb])
    target_embeddings = tf.get_variable('target_embeddings', [target_vocab_size, dim_emb])

    decoder_outputs = Build_NMT_Model(x=x, y=None, decoding_mask=None, keep_prob=keep_prob,
                                      src_embeddings=src_embeddings, target_embeddings=target_embeddings, is_train=False)

    # generated sentences
    generated = decoder_outputs.sample_id

    sess = tf.Session()

    #sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    model_path = FLAGS.save_model_path if FLAGS.fine_tune else FLAGS.pretrained_model_path

    ckpt = tf.train.latest_checkpoint(model_path)

    if ckpt:
        saver.restore(sess, ckpt)
    else:
        raise IOError("Pre-trained model is required")


    datasetGold = dict(annotations=[])
    datasetHypo = dict(annotations=[])

    cnt = 0

    while True:
        cur_batch = batcher.next_batch()

        try:
            if np.any(np.max(cur_batch['src_ixs'], axis=1)==0) or np.any(np.max(cur_batch['target_ixs'], axis=1)==0):
                continue

        except:
            import ipdb
            ipdb.set_trace()

        if batcher.epoch > 0:
            break

        eval_feed = {x: cur_batch['src_ixs'],
                     keep_prob: 1.0}

        generated_sent = sess.run(generated, feed_dict=eval_feed)

        hypo = [ ix2word['target'][i] for i in generated_sent[0] ]
        hypo_truncated = ' '.join(hypo[:np.argmax(np.array(hypo)=='<end>')])

        gold = cur_batch['target_sent'][0].split()
        gold_truncated = ' '.join(gold[:np.argmax(np.array(gold)=='<end>')])

        datasetGold['annotations'].append(dict(sentence_id=cnt, caption=gold_truncated))
        datasetHypo['annotations'].append(dict(sentence_id=cnt, caption=hypo_truncated))

        #print ' '.join(hypo)
        #print cur_batch['target_sent'][0]

        #import ipdb
        #ipdb.set_trace()


        cnt += 1
        print cnt

    json.dump(datasetGold, file('./results/gold_val.json', 'wb'))
    json.dump(datasetHypo, file('./results/hypo_val.json', 'wb'))

def demo():
    src_vocab_size = len(ix2word['src'])
    target_vocab_size = len(ix2word['target'])

    x = tf.placeholder('int32', [batch_size,None])
    keep_prob = tf.placeholder('float32')

    src_embeddings = tf.get_variable('src_embeddings', [src_vocab_size, dim_emb])
    target_embeddings = tf.get_variable('target_embeddings', [target_vocab_size, dim_emb])

    decoder_outputs = Build_NMT_Model(x=x, y=None, decoding_mask=None, keep_prob=keep_prob,
                                      src_embeddings=src_embeddings, target_embeddings=target_embeddings, is_train=False)

    # generated sentences
    generated = decoder_outputs.sample_id

    sess = tf.Session()

    saver = tf.train.Saver()

    model_path = FLAGS.save_model_path if FLAGS.fine_tune else FLAGS.pretrained_model_path

    ckpt = tf.train.latest_checkpoint(model_path)

    if ckpt:
        saver.restore(sess, ckpt)
    else:
        raise IOError("Pre-trained model is required")


    def convert_to_ids(dict, input_list):
        return [dict.get(_in, FLAGS.UNK) for _in in input_list]

    usr = raw_input("Input : ").strip().decode('utf-8')
    usr_char = list(usr.replace(u'\ufeff', '').replace(' ', ''))

    usr_char_ids = convert_to_ids(word2ix['src'], usr_char)

    eval_feed = {x: [usr_char_ids],
                 keep_prob: 1.0}

    generated_sent = sess.run(generated, feed_dict=eval_feed)

    hypo = [ ix2word['target'][i] for i in generated_sent[0] ]
    hypo_truncated = ''.join(hypo[:np.argmax(np.array(hypo)=='<end>')])
    print hypo_truncated

if __name__ == '__main__':
    if FLAGS.mode=='train':
        train()
    if FLAGS.mode=='eval':
        eval()
    if FLAGS.mode=='demo':
        demo()