import gc
import tensorflow.compat.v1 as tf
from utils import get_data_info, read_data, load_word_embeddings
from model import RAM

# args = get_args_parser().parse_args(['-model_dir', 'F:/RAM-bert/uncased_L-12_H-768_A-12',
#                                      '-pooling_strategy', 'NONE',
#                                      '-max_seq_len','85'])
#
# server = BertServer(args)
# server.start()

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 768, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 32, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_epoch', 15, 'number of epoch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('n_hop', 3, 'number of hop')
tf.app.flags.DEFINE_integer('pre_processed', 0, 'Whether the data is pre-processed')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout')
tf.app.flags.DEFINE_integer('word2id',0,'id')
tf.app.flags.DEFINE_integer('max_sentence_len',0,'max_sentence_len')
tf.app.flags.DEFINE_integer('max_aspect_len',0,'max_aspect_len')
tf.app.flags.DEFINE_float('word2vec', 0, 'word2vec')
tf.app.flags.DEFINE_float('tokenizer', 0, 'tokenizer')


# tf.app.flags.DEFINE_string('embedding_fname', 'data/vectors1.txt', 'embedding file name')
tf.app.flags.DEFINE_string('train_fname', 'data/restaurant/train.xml', 'training file name')
tf.app.flags.DEFINE_string('test_fname', 'data/restaurant/test.xml', 'testing file name')
tf.app.flags.DEFINE_string('data_info', 'data/data_info.txt', 'the file saving data information')
tf.app.flags.DEFINE_string('train_data', 'data/train_data.txt', 'the file saving training data')
tf.app.flags.DEFINE_string('test_data', 'data/test_data.txt', 'the file saving testing data')
tf.app.flags.DEFINE_string('bert_model_dir', 'bert/bert-base-uncased', '-')


def main(_):
    # tokenizer = BertTokenizer.from_pretrained(FLAGS.bert_model_dir, from_tf=True)
    # FLAGS.tokenizer = tokenizer

    print('Loading data info ...')
    FLAGS.word2id, FLAGS.max_sentence_len, FLAGS.max_aspect_len = get_data_info(FLAGS.train_fname, FLAGS.test_fname, FLAGS.data_info, FLAGS.pre_processed)
    FLAGS.max_sentence_len = FLAGS.max_sentence_len + 6
    FLAGS.max_aspect_len = FLAGS.max_aspect_len + 6
    print(FLAGS.max_sentence_len)
    # print('Loading pre-trained word vectors ...')
    # FLAGS.word2vec = load_word_embeddings(FLAGS.embedding_fname, FLAGS.embedding_dim, FLAGS.word2id)

    print('Loading training data and testing data ...')
    train_data = read_data(FLAGS.train_fname, FLAGS.word2id, FLAGS.max_sentence_len, FLAGS.max_aspect_len, FLAGS.train_data, FLAGS.pre_processed)
    test_data = read_data(FLAGS.test_fname, FLAGS.word2id, FLAGS.max_sentence_len,  FLAGS.max_aspect_len, FLAGS.test_data, FLAGS.pre_processed)

    # config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        model = RAM(FLAGS, sess)
        model.build_model()
        model.run(train_data, test_data)
    del train_data, test_data
    gc.collect()



if __name__ == '__main__':
    tf.app.run()

