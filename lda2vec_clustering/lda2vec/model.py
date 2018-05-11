from __future__ import division #py2
from datetime import datetime
import os
import os.path
import sys
import pickle
import time
import shelve

import errno #py2

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from lda2vec import dirichlet_likelihood
from lda2vec import EmbedMixture
from lda2vec import NegativeSampling
from lda2vec import utils
from lda2vec import corpus
from lda2vec import prepare_topics, print_top_words_per_topic, topic_coherence

corpus = corpus.Corpus

LOG_DIR = 'logs'

class LDA2Vec():

    DEFAULTS = {
        "n_document_topics": 15,
        "n_embedding": 300, # embedding size

        "batch_size": 20,
        "window": 5,
        "learning_rate": 1E-3,
        "dropout_ratio": 0.8, # keep_prob
        "word_dropout": 0.8, #1.

        "power": 0.75, # unigram sampler distortion
        "n_samples": 50, # num negative samples

        "temperature": 1., # embed mixture temp
        "lmbda": 200., # strength of Dirichlet prior
        "alpha": None, # alpha of Dirichlet process (defaults to 1/n_topics)
    }
    RESTORE_KEY = "to_restore"

    def __init__(self, n_documents=None, n_vocab=None, d_hyperparams={},
                 freqs=None, w_in=None, fixed_words=False, word2vec_only=False,
                 meta_graph=None, save_graph_def=True, log_dir="./log"):
        """
        Initialize LDA2Vec model from parameters or saved `meta_graph`

        Args:
            n_documents (int)
            n_vocab (int)
            d_hyperparams (dict): model hyperparameters

            freqs (list or np.array): iterable of word frequencies for candidate sampler
                                      (None defaults to unigram sampler)
            w_in (np.array): pre-trained word embeddings (n_vocab x n_embedding)

            fixed_words (bool): train doc and topic weights only?
            word2vec_only (bool): word2vec context and objective only?

            meta_graph (str): path/to/saved/meta_graph (without `.meta`)

            save_graph_def (bool): save graph_def to file?
            log_dir (str): path/to/logging/outputs

        """

        assert not (n_documents is None and n_vocab is None and meta_graph is None), (
                "Must initialize new model or pass saved meta_graph")
        assert not (fixed_words and w_in is None), (
                "If fixing words, must supply pre-trained word embeddings")
        assert not (fixed_words and word2vec_only), ("Nothing to train here...")

        self.__dict__.update(LDA2Vec.DEFAULTS, **d_hyperparams)
        
        tf.reset_default_graph()
        self.sesh = tf.Session()
        self.moving_avgs = tf.train.ExponentialMovingAverage(0.9)

        if not meta_graph: # new model
            self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")

            # build graph
            self.mixture = EmbedMixture(
                    n_documents, self.n_document_topics, self.n_embedding,
                    temperature=self.temperature)

            # optionally, pass in pre-trained non/trainable word embeddings
            if w_in is not None:
                assert n_vocab == w_in.shape[0], "Word embeddings must match vocab size"
            W_in = (w_in if w_in is None else (tf.constant(w_in) if fixed_words
                                               else tf.Variable(w_in)))
            self.sampler = NegativeSampling(
                    self.n_embedding, n_vocab, self.n_samples, power=self.power,
                    freqs=freqs, W_in=W_in)

            handles = self._buildGraph(word2vec_only=word2vec_only) + (
                self.mixture(), self.mixture.proportions(softmax=True),
                self.mixture.factors, self.sampler.W)

            for handle in handles:
                tf.add_to_collection(LDA2Vec.RESTORE_KEY, handle)
                self.sesh.run(tf.global_variables_initializer()) #py2 to change ???

        else: # restore saved model
            datetime_prev, _ = os.path.basename(meta_graph).split("_lda2vec")
            datetime_now = datetime.now().strftime(r"%y%m%d_%H%M")
            self.datetime = "{}_{}".format(datetime_prev, datetime_now)

            # rebuild graph
            meta_graph = os.path.abspath(meta_graph)
            tf.train.import_meta_graph(meta_graph + ".meta").restore(
                self.sesh, meta_graph)
            handles = self.sesh.graph.get_collection(LDA2Vec.RESTORE_KEY)

        # unpack tensor ops to feed or fetch
        (self.pivot_idxs, self.doc_at_pivot, self.dropout, self.target_idxs,
         self.fraction, self.loss_word2vec, self.loss_lda, self.loss,
         self.global_step, self.train_op, self.switch_loss, self.doc_embeds,
         self.doc_proportions, self.topics, self.word_embeds) = handles

        self.log_dir = "{}_{}".format(log_dir, self.datetime)
        if save_graph_def: # tensorboard
            self.logger = tf.summary.FileWriter(self.log_dir, self.sesh.graph)


    @property
    def step(self):
        """Train step"""
        return self.sesh.run(self.global_step)


    def _buildGraph(self, word2vec_only=False):

        self.global_step = tf.Variable(0, trainable=False)

        # pivot word
        self.pivot_idxs = tf.placeholder(tf.int32,
                                    shape=[None,], # None enables variable batch size
                                    name="pivot_idxs")
        pivot = tf.nn.embedding_lookup(self.sampler.W, self.pivot_idxs) # word embedding

        # doc
        self.doc_at_pivot = tf.placeholder(tf.int32, shape=[None,], name="doc_ids")
        doc = self.mixture(self.doc_at_pivot) # doc embedding

        # context is sum of doc (mixture projected onto topics) & pivot embedding
        self.dropout = self.mixture.dropout
        self.switch_loss = tf.Variable(0, trainable=False)

        # context 
        contexts = (tf.nn.dropout(pivot, self.dropout), tf.nn.dropout(doc, self.dropout))
        context = (tf.cond(self.global_step < self.switch_loss,
                          lambda: contexts[0],
                          lambda: tf.add(*contexts)) if not word2vec_only
                   else contexts[0])

        # targets
        self.target_idxs = tf.placeholder(tf.int64, shape=[None,], name="target_idxs")

        # NCE loss
        with tf.name_scope("nce_loss"):
            self.loss_word2vec = self.sampler(context, self.target_idxs)
            self.loss_word2vec = utils.print_(self.loss_word2vec, "loss_word2vec")

        # dirichlet loss (proportional to minibatch fraction)
        with tf.name_scope("lda_loss"):
            self.fraction = tf.Variable(1, trainable=False, dtype=tf.float32)
            self.loss_lda = self.lmbda * self.fraction * self.prior() # dirichlet log-likelihood
            self.loss_lda = utils.print_(self.loss_lda, "loss_lda")

        # optimize
        # loss = tf.identity(loss_word2vec + self.lmbda * loss_lda, "loss")
        # loss = tf.identity(loss_word2vec + loss_lda, "loss")
        self.loss = (tf.cond(self.global_step < self.switch_loss,
                       lambda: self.loss_word2vec,
                       lambda: self.loss_word2vec + self.loss_lda) if not word2vec_only
                else tf.identity(self.loss_word2vec)) # avoid duplicating moving avg (ValueError)

        loss_avgs_op = self.moving_avgs.apply([self.loss_lda, self.loss_word2vec, self.loss])

        with tf.control_dependencies([loss_avgs_op]):
            self.train_op = tf.contrib.layers.optimize_loss(
                    self.loss, self.global_step, self.learning_rate, 'Adam', clip_gradients=5.)

        return (self.pivot_idxs, self.doc_at_pivot, self.dropout, self.target_idxs, self.fraction,
                self.loss_word2vec, self.loss_lda, self.loss, self.global_step, self.train_op, self.switch_loss)


    def prior(self):
        # defaults to inialization with uniform prior (1/n_topics)
        return dirichlet_likelihood(self.mixture.W, alpha=self.alpha)


    def _addSummaries(self, metadata="metadata.tsv",
                      metadata_docs="metadata.docs.tsv"):
        # summary nodes
        tf.summary.scalar("loss_lda", self.loss_lda)
        tf.summary.scalar("loss_nce", self.loss_word2vec)

        tf.summary.scalar("loss_lda_avg", self.moving_avgs.average(self.loss_lda))
        tf.summary.scalar("loss_nce_avg", self.moving_avgs.average(self.loss_word2vec))
        tf.summary.scalar("loss_avg", self.moving_avgs.average(self.loss))

        tf.summary.histogram("word_embeddings_hist", self.word_embeds)
        tf.summary.histogram("topic_embeddings_hist", self.topics)
        tf.summary.histogram("doc_embeddings_hist", self.doc_embeds)

        tf.summary.scalar("doc_mixture_sparsity",
                          tf.nn.zero_fraction(self.doc_proportions))

        # VIZ
        config = projector.ProjectorConfig()
        
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = self.word_embeds.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = os.path.join(self.log_dir, metadata)

        topic_embedding = config.embeddings.add()
        topic_embedding.tensor_name = self.topics.name

        doc_embedding = config.embeddings.add()
        doc_embedding.tensor_name = self.doc_embeds.name
        doc_embedding.metadata_path = os.path.join(self.log_dir, metadata_docs)

        doc_props = config.embeddings.add()
        doc_props.tensor_name = self.doc_proportions.name
        doc_props.metadata_path = os.path.join(self.log_dir, metadata_docs)
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(self.logger, config)

        return tf.summary.merge_all()


    def make_feed_dict(self, doc_ids, word_indices, window=None,
                       update_only_docs=False):

        window = (self.window if window is None else window)

        if len(doc_ids)< 2*window :    
            doc_at_pivot = doc_ids
            pivot_idx = word_indices
        else : 
            doc_at_pivot = doc_ids[window: -window]
            pivot_idx = word_indices[window: -window]

        start, end = window, word_indices.shape[0] - window

        target_idxs = []

        for frame in range(-window, window + 1):

            # Skip predicting the current pivot
            if frame == 0:
                continue

            # Predict word given context and pivot word
            # The target starts before the pivot
            
            if len(doc_ids)< 2*window : 
                if len(doc_ids) < 2 : 
                    continue
                
                doc_at_target = doc_ids
                target_idx = word_indices
            else : 
                doc_at_target = doc_ids[start + frame: end + frame]
                target_idx = word_indices[start + frame: end + frame]
            
            doc_is_same = doc_at_target == doc_at_pivot 


            rand = np.random.uniform(0, 1, doc_is_same.shape[0])
            mask = (rand < self.word_dropout)
            weight = np.logical_and(doc_is_same, mask).astype(np.int32)

            # If weight is 1.0 then targetidx
            # If weight is 0.0 then -1
            target_idx = target_idx * weight + -1 * (1 - weight)

            target_idxs.append(target_idx)

        pivot_idxs = np.tile(pivot_idx, window * 2)
        docs_at_pivot = np.tile(doc_at_pivot, window * 2)
        target_idxs = np.concatenate(target_idxs)

        # ignore training points due to OOV or dropout
        # TODO set OOV token globally
        LAST_OOV_TOKEN = 1
        # mask = np.logical_and((target_idxs > 0), (pivot_idxs > 0))
        mask = np.logical_and((target_idxs > LAST_OOV_TOKEN),
                              (pivot_idxs > LAST_OOV_TOKEN))
        mask = mask.tolist()
        # assert sum(mask) > 0, "At least one example must not be masked"

        feed_dict = {self.pivot_idxs: pivot_idxs[mask],
                     self.doc_at_pivot: docs_at_pivot[mask],
                     self.target_idxs: target_idxs[mask],
                     self.dropout: self.dropout_ratio}

        return feed_dict



    def train(self, doc_ids, flattened, vocab, words, max_epochs=np.inf, verbose=False,  #added vocab & words to add save(npz) 
                                                                                         #option during training
              loss_switch_epochs=0, # num epochs until LDA loss switched on
              save=False, save_every=1000, outdir="./out", summarize=True,
              summarize_every=1000, metadata="metadata.tsv",
              metadata_docs="metadata.docs.tsv"):


        n_vocab = flattened.max() + 1
        # How many tokens are in each document
        doc_idx, lengths = np.unique(doc_ids, return_counts=True)
        doc_lengths = np.zeros(doc_ids.max() + 1, dtype='int32')
        doc_lengths[doc_idx] = lengths
        # Count all token frequencies
        tok_idx, freq = np.unique(flattened, return_counts=True)
        term_frequency = np.zeros(n_vocab, dtype='int32')
        term_frequency[tok_idx] = freq

        if save:
            try:
                os.mkdir(outdir)
            except OSError as e:    #for Python 2
                if e.errno == errno.EEXIST:
                    pass
            saver = tf.train.Saver(tf.global_variables())
            outdir = os.path.abspath(self.log_dir)

        if summarize:
            try:
                self.logger.flush()
            except(AttributeError): # not yet logging
                self.logger = tf.summary.FileWriter(self.log_dir, self.sesh.graph)
            merged = self._addSummaries(metadata, metadata_docs)

        j = 0
        epoch = 0

        fraction = self.batch_size / len(flattened) # == batch / n_corpus
        self.sesh.run(tf.assign(self.fraction, fraction))
        progress = shelve.open('progress.shelve')

        # turn on LDA loss after n iters of training
        iters_per_epoch = (int(len(flattened) / self.batch_size) +
                           np.ceil(len(flattened) % self.batch_size))
        n = iters_per_epoch * loss_switch_epochs
        self.sesh.run(tf.assign(self.switch_loss, n))

        now = datetime.now().isoformat()[11:]
        print("------- Training begin: {} -------\n".format(now))

        while epoch < max_epochs:
            try:

                # doc_ids, word_idxs
                for d, f in utils.chunks(self.batch_size, doc_ids, flattened):
                    t0 = datetime.now()

                    feed_dict = self.make_feed_dict(d, f)

                    # if len(feed_dict[self.pivot_idxs]) == 0:
                    # 	print("Empty batch. Skipping...")
                    # 	continue

                    fetches = [self.loss_lda, self.loss_word2vec,
                               self.loss, self.train_op]
                    loss_lda, loss_word2vec, loss, _ = self.sesh.run(
                        fetches, feed_dict=feed_dict)

                    if j > 5 : 
                        print(loss_lda, loss_word2vec, loss) #py2

                    

                    j += 1

                    if verbose and j % 1000 == 0:
                        msg = ("J:{j:05d} E:{epoch:05d} L_nce:{l_word2vec:1.3e} "
                               "L_dirichlet:{l_lda:1.3e} R:{rate:1.3e}")

                        t1 = datetime.now().timestamp()
                        dt = t1 - t0
                        rate = self.batch_size / dt
                        logs = dict(l_word2vec=loss_word2vec, epoch=epoch, j=j,
                                    l_lda=loss_lda, rate=rate)

                        print(msg.format(**logs))

                    if save and j % save_every == 0:
                        outfile = os.path.join(outdir,
                                               "{}_lda2vec".format(self.datetime))
                        saver.save(self.sesh, outfile, global_step=self.step)

                    if summarize and j % summarize_every == 0:
                        summary = self.sesh.run(merged, feed_dict=feed_dict)
                        self.logger.add_summary(summary, global_step=self.step)

                    #if j % 100 == 0 and j > 100 and epoch > 1:
                    #    coherence = topic_coherence(top_words)
                    #    for j in range(n_topics):
                    #        print(j, coherence[(j, 'cv')])
                    #    kw = dict(top_words=top_words, coherence=coherence, epoch=epoch)
                    #    progress[str(epoch)] = pickle.dumps(kw)
                    
                epoch += 1
                a = self.mixture.W.eval(session=self.sesh)
                b = self.mixture.factors.eval(session=self.sesh)
                c = self.sampler.W.eval(session=self.sesh)
                data = prepare_topics(a, b, c, words)
                print("------- epoch: {}-------\n".format(epoch))
                top_words = print_top_words_per_topic(data)
                data['doc_lengths'] = doc_lengths
                data['term_frequency'] = term_frequency
                np.savez('topics.pyldavis', **data)

            except(KeyboardInterrupt):
                break

        print("epoch", epoch)
        print("max", max_epochs)
        now = datetime.now().isoformat()[11:]
        print("------- Training end: {} -------\n".format(now))

        if save:
            outfile = os.path.join(outdir, "{}_lda2vec".format(self.datetime))
            saver.save(self.sesh, outfile, global_step=self.step)

        try:
            self.logger.flush()
            self.logger.close()
        except(AttributeError): # not logging
            pass

        #sys.exit(0) may be not needed if finite number of epochs


    def _buildGraph_similarity(self):
        """Build nodes to compute the cosine similarity between examples
        (doc/word/topic idxs) and corresponding embeddings
        """
        idxs_in = tf.placeholder(tf.int32,
                              shape=[None,], # None enables variable batch size
                              name="idxs") # doc or topic or word

        n = tf.placeholder_with_default(10, shape=None, name="n")

        normalized_embedding = dict()
        for name, embedding in zip(("word", "topic", "doc"),
                                   (self.word_embeds, self.topics, self.doc_embeds)):
            norm = tf.sqrt(tf.reduce_sum(embedding**2, 1, keep_dims=True))
            normalized_embedding[name] = embedding / norm

        similarities = dict()
        for in_, vs in (("word", "word"),
                        ("word", "topic"),
                        ("topic", "word"),
                        ("doc", "doc")):
            embeddings_in = tf.nn.embedding_lookup(normalized_embedding[in_],
                                                   idxs_in)
            similarity = tf.matmul(embeddings_in, normalized_embedding[vs],
                                   transpose_b=True)
            values, top_idxs = tf.nn.top_k(similarity, sorted=True, k=n)
            # top_sims = tf.gather_nd(similarity, top_idxs)
            # similarities[(in_, vs)] = [top_idxs, top_sims]
            similarities[(in_, vs)] = (top_idxs, similarity)

        return (idxs_in, n, similarities)


    def compute_similarity(self, ids, in_, vs, n=10):
        """Compute the cosine similarity between minibatch examples
        and all embeddings.

        Args: ids (1-D array of idxs)
              in_ = "doc" or "word" or "topic" (corresponding to ids)
              vs = "doc" or "word" or "topic" (corresponding to embedding to compare)
        """
        while True:
            try:
                feed_dict = {self.idxs_in: ids, self.n: n}
                fetches = self.similarities[(in_, vs)]
                top_idxs, sims = self.sesh.run(fetches, feed_dict=feed_dict)
                top_sims = sims[ # select similarity to top matching idxs per id
                    tuple([i]*top_idxs.shape[1] for i in range(top_idxs.shape[0])),
                     top_idxs]
                return (top_idxs, top_sims)

            except(AttributeError): # not yet initialized
                (self.idxs_in, self.n,
                 self.similarities) = self._buildGraph_similarity()


    # def validate(self, doc_ids, flattened, save=False): TODO ?

    # 	loss_word2vec = self.fit_partial(doc_ids, flattened)

    # 	loss_lda = self.sesh.run(self.prior(), feed_dict=feed_dict)

    # 	loss = loss_word2vec + self.lmbda * loss_lda

    # 	if verbose and j % 1000 == 0:
    # 			# msg = ("J:{j:05d} E:{epoch:05d} L:{loss:1.3e} " # 			# 	"P:{prior:1.3e} R:{rate:1.3e}")
    # 			msg = ("J:{j:05d} E:{epoch:05d} L_nce:{l_word2vec:1.3e} "
    # 					"L_dirichlet:{l_lda:1.3e} R:{rate:1.3e}")

    # 			t1 = datetime.now().timestamp()
    # 			dt = t1 - t0
    # 			rate = self.batch_size / dt
    # 			logs = dict(l_word2vec=loss_word2vec, epoch=epoch, j=j,
    # 					l_lda=loss_lda, rate=rate)

    # 			print(msg.format(**logs))

    def load(self, path):
        sess = tf.Session()
        # First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph('./log_180503_1444/180503_1444_lda2vec-15403100.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./log_180503_1444/'))