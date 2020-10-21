import logging,code
import keras.backend as K
from keras.layers import Dense, Dropout, Activation, Embedding, Input, Concatenate, Lambda
from keras.models import Model, Sequential
from my_layers import Conv1DWithMasking, Remove_domain_emb, Self_attention, Attention, WeightedSum
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Custom CNN kernel initializer
# Use the initialization from Kim et al. (2014) for CNN kernel
def my_init(shape, dtype=K.floatx()):
    return 0.01 * np.random.standard_normal(size=shape)

def create_model(args, vocab, nb_class, overall_maxlen, doc_maxlen_1, doc_maxlen_2):

    # Funtion that initializes word embeddings 
    def init_emb(emb_matrix, vocab, emb_file_gen, emb_file_domain):

        print 'Loading pretrained general word embeddings and domain word embeddings ...'

        counter_gen = 0.
        pretrained_emb = open(emb_file_gen)
        for line in pretrained_emb:
            tokens = line.split()
            if len(tokens) != 301:
                continue
            word = tokens[0]
            vec = tokens[1:]
            try:
                emb_matrix[0][vocab[word]][:300] = vec
                counter_gen += 1
            except KeyError:
                pass

        if args.use_domain_emb:
            counter_domain = 0.
            pretrained_emb = open(emb_file_domain)
            for line in pretrained_emb:
                tokens = line.split()
                if len(tokens) != 101:
                    continue
                word = tokens[0]
                vec = tokens[1:]
                try:
                    emb_matrix[0][vocab[word]][300:] = vec
                    counter_domain += 1
                except KeyError:
                    pass

        pretrained_emb.close()
        logger.info('%i/%i word vectors initialized by general embeddings (hit rate: %.2f%%)' % (counter_gen, len(vocab), 100*counter_gen/len(vocab)))
        
        if args.use_domain_emb:
            logger.info('%i/%i word vectors initialized by domain embeddings (hit rate: %.2f%%)' % (counter_domain, len(vocab), 100*counter_domain/len(vocab)))

        return emb_matrix


    # Build model
    logger.info('Building model ...')
    print 'Building model ...'
    print '\n\n'

    vocab_size = len(vocab)

    ###################################
    # Inputs 
    ###################################
    print 'Input layer'
    # sequence of token indices for aspect-level data
    sentence_input = Input(shape=(overall_maxlen,), dtype='int32', name='sentence_input')
    # gold opinion label for aspect-level data. 
    op_label_input = Input(shape=(overall_maxlen, 3), dtype=K.floatx(), name='op_label_input')
    # probability of sending gold opinion labels at opinion transmission step
    p_gold_op = Input(shape=(overall_maxlen,), dtype=K.floatx(), name='p_gold_op') 

    A_in = Input(shape=(overall_maxlen, overall_maxlen), dtype=K.floatx(), name='A_input')

    if args.use_doc:
        # doc_input_1 denotes the data for sentiment classification
        # doc_input_2 denotes the data for domain classification
        doc_input_1 = Input(shape=(doc_maxlen_1,), dtype='int32', name='doc_input_1')
        doc_input_2 = Input(shape=(doc_maxlen_2,), dtype='int32', name='doc_input_2')

    if args.use_bert:
        bert_input = Input(shape=(overall_maxlen + 1, 768), dtype=K.floatx(), name='bert_input') # +1 denote +cls
    #########################################
    # Shared word embedding layer 
    #########################################
    print 'Word embedding layer'
    word_emb = Embedding(vocab_size, args.emb_dim, mask_zero=True, name='word_emb')

    # aspect-level inputs
    word_embeddings = word_emb(sentence_input) 
    sentence_output = word_embeddings

    # doc-level inputs
    if args.use_doc:
        doc_output_1 = word_emb(doc_input_1)
        # we only use general embedding for domain classification
        doc_output_2 = word_emb(doc_input_2)
        if args.use_domain_emb:
            # mask out the domain embeddings
            doc_output_2 = Remove_domain_emb()(doc_output_2)

    def slice(x, index):
        return x[:, index, :]
    def slice1(x, index):
        return x[:, index:, :]
    expand_dim = Lambda(lambda x: K.expand_dims(x, axis = 1))
    if args.use_bert:
        #code.interact(local=locals())
        bert_inp = Lambda(slice1, arguments={'index': 1})(bert_input)
        bert_cls = Lambda(slice, arguments={'index': 0})(bert_input)
        #sentence_output = Concatenate()([sentence_output, bert_inp])
#       if args.use_bert_cls:
     #code.interact(local=locals())
         #bert_cls = bert_input[:,0,:]
        node_num = sentence_output.shape.as_list()[1]
        bert_cls1 = expand_dim(bert_cls)
        bert_cls = Lambda(lambda x: K.tile(x, [1, node_num, 1]))(bert_cls1)


    from my_layers_algo import DigiCaps, Length, Capsule

    if args.use_bert_cls == 0 and args.use_bert:
        sentence_output = Concatenate()([sentence_output, bert_cls])

    ######################################
    # Shared CNN layers
    ######################################

    for i in xrange(args.shared_layers):
        print 'Shared CNN layer %s'%i
        sentence_output = Dropout(args.dropout_prob)(sentence_output)
        if args.use_doc:
            doc_output_1 = Dropout(args.dropout_prob)(doc_output_1)
            doc_output_2 = Dropout(args.dropout_prob)(doc_output_2)

        if i == 0:
            #conv_0 = Conv1DWithMasking(filters=args.cnn_dim/2, kernel_size=2, \
            #  activation='relu', padding='same', kernel_initializer=my_init, name='cnn_0_0')
            conv_1 = Conv1DWithMasking(filters=args.cnn_dim/2, kernel_size=3, \
              activation='relu', padding='same', kernel_initializer=my_init, name='cnn_0_1')
            #conv_2 = Conv1DWithMasking(filters=args.cnn_dim/2, kernel_size=4, \
            #  activation='relu', padding='same', kernel_initializer=my_init, name='cnn_0_2')
            conv_3 = Conv1DWithMasking(filters=args.cnn_dim/2, kernel_size=5, \
              activation='relu', padding='same', kernel_initializer=my_init, name='cnn_0_3')

            #sentence_output_0 = conv_0(sentence_output)
            sentence_output_1 = conv_1(sentence_output)
            #sentence_output_2 = conv_2(sentence_output)
            sentence_output_3 = conv_3(sentence_output)
            #sentence_output = Concatenate()([sentence_output_0, sentence_output_1, sentence_output_2, sentence_output_3])
            sentence_output = Concatenate()([sentence_output_1, sentence_output_3])

            if args.use_doc:

                #doc_output_1_0 = conv_0(doc_output_1)
                doc_output_1_1 = conv_1(doc_output_1)
                #doc_output_1_2 = conv_2(doc_output_1)
                doc_output_1_3 = conv_3(doc_output_1)
                #doc_output_1 = Concatenate()([doc_output_1_0, doc_output_1_1, doc_output_1_2, doc_output_1_3])
                doc_output_1 = Concatenate()([doc_output_1_1, doc_output_1_3])

                #doc_output_2_0 = conv_0(doc_output_2)
                doc_output_2_1 = conv_1(doc_output_2)
                #doc_output_2_2 = conv_2(doc_output_2)
                doc_output_2_3 = conv_3(doc_output_2)
                #doc_output_2 = Concatenate()([doc_output_2_0, doc_output_2_1, doc_output_2_2, doc_output_2_3])
                doc_output_2 = Concatenate()([doc_output_2_1, doc_output_2_3])

        else:
            #conv = Conv1DWithMasking(filters=args.cnn_dim, kernel_size=3, \
            #  activation='relu', padding='same', kernel_initializer=my_init, name='cnn_3_%s'%i)
            conv_ = Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
              activation='relu', padding='same', kernel_initializer=my_init, name='cnn_5_%s'%i)
            #sentence_output1 = conv(sentence_output)
            sentence_output = conv_(sentence_output)
            #sentence_output = Concatenate()([sentence_output1, sentence_output2])

            if args.use_doc:
                doc_output_1 = conv_(doc_output_1)
                
                doc_output_2 = conv_(doc_output_2)
                

        word_embeddings = Concatenate()([word_embeddings, sentence_output])

    init_shared_features = sentence_output


    #######################################
    # Define task-specific layers
    #######################################
    #if args.which_dual == 'dual':
     #   from my_layers import Conv1DWithMasking, Remove_domain_emb, Self_attention, Attention, WeightedSum, Dual_attention
    #else:
   #     from my_layers_algo import Conv1DWithMasking, Remove_domain_emb, Self_attention, Attention, WeightedSum, Dual_attention
    # AE specific layers
    aspect_cnn = Sequential()
    for a in xrange(args.aspect_layers):
        print 'Aspect extraction layer %s'%a
        aspect_cnn.add(Dropout(args.dropout_prob))
        aspect_cnn.add(Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
                  activation='relu', padding='same', kernel_initializer=my_init, name='aspect_cnn_%s'%a))
    aspect_dense = Dense(nb_class, activation='softmax', name='aspect_dense')

    # OE specific layers
    opinion_cnn = Sequential()
    for a in xrange(args.opinion_layers):
        print 'Opinion extraction layer %s'%a
        opinion_cnn.add(Dropout(args.dropout_prob))
        opinion_cnn.add(Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
                  activation='relu', padding='same', kernel_initializer=my_init, name='opinion_cnn_%s'%a))
    opinion_dense = Dense(nb_class, activation='softmax', name='opinion_dense')

    # AS specific layers
    sentiment_cnn = Sequential()
    for b in xrange(args.senti_layers):
        print 'Sentiment classification layer %s'%b
        sentiment_cnn.add(Dropout(args.dropout_prob))
        sentiment_cnn.add(Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
                  activation='relu', padding='same', kernel_initializer=my_init, name='sentiment_cnn_%s'%b))

    sentiment_att = Self_attention(args.use_opinion, name='sentiment_att')
    sentiment_dense = Dense(3, activation='softmax', name='sentiment_dense')

    aspect_dual_att = Dual_attention(name='aspect_dualatt')
    opinion_dual_att = Dual_attention(name='opinion_dualatt')
    sentiment_dual_att = Dual_attention(name='sentiment_dualatt')

    asp_caps = Capsule(num_capsule=overall_maxlen, A=A_in, dim_capsule=args.capsule_dim, routings=3, name='asp_caps')
    senti_caps = Capsule(num_capsule=overall_maxlen, A=A_in, dim_capsule=args.capsule_dim, routings=3, name='senti_caps')
    opin_caps = Capsule(num_capsule=overall_maxlen, A=A_in, dim_capsule=args.capsule_dim, routings=3, name='opin_caps')
    
    #probs = Length(name='out_caps')
    if args.use_doc:
        # DS specific layers
        doc_senti_cnn = Sequential()
        for c in xrange(args.doc_senti_layers):
            print 'Document-level sentiment layers %s'%c
            doc_senti_cnn.add(Dropout(args.dropout_prob))
            doc_senti_cnn.add(Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
                      activation='relu', padding='same', kernel_initializer=my_init, name='doc_sentiment_cnn_%s'%c))

        doc_senti_att = Attention(name='doc_senti_att')
        doc_senti_dense = Dense(3, name='doc_senti_dense')
        # The reason not to use the default softmax is that it reports errors when input_dims=2 due to 
        # compatibility issues between the tf and keras versions used.
        softmax = Lambda(lambda x: K.tf.nn.softmax(x), name='doc_senti_softmax')

        # DD specific layers
        doc_domain_cnn = Sequential()
        for d in xrange(args.doc_domain_layers):
            print 'Document-level domain layers %s'%d 
            doc_domain_cnn.add(Dropout(args.dropout_prob))
            doc_domain_cnn.add(Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
                      activation='relu', padding='same', kernel_initializer=my_init, name='doc_domain_cnn_%s'%d))

        doc_domain_att = Attention(name='doc_domain_att')
        doc_domain_dense = Dense(1, activation='sigmoid', name='doc_domain_dense')

    # re-encoding layer
    enc = Dense(args.cnn_dim, activation='relu', name='enc')
    enc_a =Dense(args.cnn_dim, activation='relu', name='enc_a')
    enc_o =Dense(args.cnn_dim, activation='relu', name='enc_o')
    enc_s =Dense(args.cnn_dim, activation='relu', name='enc_s')
    enc_d = Dense(args.cnn_dim, activation='relu', name='enc_d')

    ####################################################
    # aspect-level operations involving message passing
    ####################################################
    print(sentence_output)
#    sentence_output = enc(sentence_output)
    aspect_output = sentence_output
    opinion_output = sentence_output
    sentiment_output = sentence_output

    doc_senti_output = sentence_output
    doc_domain_output = sentence_output
    for i in xrange(args.interactions+1):
        print 'Interaction number ', i
        if args.use_doc:
            ### DS ###
            if args.doc_senti_layers > 0:
                doc_senti_output = doc_senti_cnn(doc_senti_output)
            # output attention weights with two activation functions
            senti_att_weights_softmax, senti_att_weights_sigmoid = doc_senti_att(doc_senti_output)

            # reshape the sigmoid attention weights, will be used in message passing
            senti_weights = Lambda(lambda x: K.expand_dims(x, axis=-1))(senti_att_weights_sigmoid)
            doc_senti_output1 = WeightedSum()([doc_senti_output, senti_att_weights_softmax])
            doc_senti_output1 = Dropout(args.dropout_prob)(doc_senti_output1)
            doc_senti_output1 = doc_senti_dense(doc_senti_output1)
            doc_senti_probs = softmax(doc_senti_output1)
            # reshape the doc-level sentiment predictions, will be used in message passing
            doc_senti_probs = Lambda(lambda x: K.expand_dims(x, axis=-2))(doc_senti_probs)
            doc_senti_probs = Lambda(lambda x: K.repeat_elements(x, overall_maxlen, axis=1))(doc_senti_probs)

             ### DD ###
            if args.doc_domain_layers > 0:
                doc_domain_output = doc_domain_cnn(doc_domain_output)
            domain_att_weights_softmax, domain_att_weights_sigmoid = doc_domain_att(doc_domain_output)
            domain_weights = Lambda(lambda x: K.expand_dims(x, axis=-1))(domain_att_weights_sigmoid)
            #code.interact(local=locals())
            doc_domain_output1 = WeightedSum()([doc_domain_output, domain_att_weights_softmax])
            doc_domain_output1 = Dropout(args.dropout_prob)(doc_domain_output1)
            doc_domain_probs = doc_domain_dense(doc_domain_output1)

        if args.use_bert:
            aspect_output = Concatenate()([aspect_output, bert_inp])
            opinion_output = Concatenate()([opinion_output, bert_inp])
            sentiment_output = Concatenate()([sentiment_output, bert_inp])
            aspect_output = Dropout(args.dropout_prob)(aspect_output)
            opinion_output = Dropout(args.dropout_prob)(opinion_output)
            sentiment_output = Dropout(args.dropout_prob)(sentiment_output)

        ### AE ###
        if args.aspect_layers > 0:
            aspect_output = aspect_cnn(aspect_output)
        # concate word embeddings and task-specific output for prediction
         ### OE ###
        if args.aspect_layers > 0:
            opinion_output = opinion_cnn(opinion_output)
         ### AS ###
        if args.senti_layers > 0:
            sentiment_output = sentiment_cnn(sentiment_output)

        opin2asp = asp_caps([aspect_output, opinion_output])
        senti2asp = asp_caps([aspect_output, sentiment_output])
        asp = Concatenate()([opin2asp, senti2asp])

        asp2opin = opin_caps([opinion_output, aspect_output])
        senti2opin = opin_caps([opinion_output, sentiment_output])
        opin = Concatenate()([asp2opin, senti2opin])

        asp2senti = senti_caps([sentiment_output, aspect_output])
        opin2senti = senti_caps([sentiment_output, opinion_output])
        senti = Concatenate()([asp2senti, opin2senti])
        #sentiment_output = sentiment_att([sentiment_output, op_label_input, opinion_probs, p_gold_op])

#        aspect_output += asp
#        opinion_output += opin
#        sentiment_output += senti
        if args.use_doc:
            aspect_output = Concatenate()([word_embeddings, aspect_output, asp, domain_weights])
            opinion_output = Concatenate()([word_embeddings, opinion_output, opin, domain_weights])
            sentiment_output = Concatenate()([init_shared_features, sentiment_output, senti, doc_senti_probs, senti_weights])
        else:
            aspect_output = Concatenate()([word_embeddings, aspect_output, asp])
            opinion_output = Concatenate()([word_embeddings, opinion_output, opin])
            sentiment_output = Concatenate()([init_shared_features, sentiment_output, senti])
        #aspect_output = Concatenate()([init_shared_features, aspect_output])
        aspect_output = Dropout(args.dropout_prob)(aspect_output)
        aspect_probs = aspect_dense(aspect_output)

        #opinion_output = Concatenate()([init_shared_features, opinion_output])
        opinion_output = Dropout(args.dropout_prob)(opinion_output)
        opinion_probs = opinion_dense(opinion_output)

        #sentiment_output = Concatenate()([word_embeddings, sentiment_output])
        sentiment_output = Dropout(args.dropout_prob)(sentiment_output)
        sentiment_probs = sentiment_dense(sentiment_output)
        
        # update sentence_output for the next iteration
        
        opinion_output = Concatenate()([opinion_output, aspect_probs, opinion_probs, sentiment_probs, domain_weights])
        aspect_output = Concatenate()([aspect_output, aspect_probs, opinion_probs, sentiment_probs, domain_weights])
        sentiment_output = Concatenate()([sentiment_output, aspect_probs, opinion_probs, sentiment_probs, doc_senti_probs, senti_weights])
        sentence_output_ = Concatenate()([sentence_output, aspect_probs, opinion_probs, sentiment_probs,
                                                doc_senti_probs, senti_weights, domain_weights])       
        #code.interact(local=locals())
        aspect_output = enc_a(aspect_output)
        opinion_output = enc_o(opinion_output)
        sentiment_output = enc_s(sentiment_output)
        if args.use_doc:
            doc_senti_output = enc_d(sentence_output_)
            doc_domain_output = enc_d(sentence_output_)
        if args.use_bert:
            aspect_model = Model(inputs=[sentence_input, A_in, op_label_input, p_gold_op, bert_input], outputs=[aspect_probs, opinion_probs, sentiment_probs])
        else:
            aspect_model = Model(inputs=[sentence_input, A_in, op_label_input, p_gold_op], outputs=[aspect_probs, opinion_probs, sentiment_probs])


    ####################################################
    # doc-level operations without message passing
    ####################################################

    if args.use_doc:
        if args.doc_senti_layers > 0:
            doc_output_1 = doc_senti_cnn(doc_output_1)
        att_1, _ = doc_senti_att(doc_output_1)
        doc_output_1 = WeightedSum()([doc_output_1, att_1])
        doc_output_1 = Dropout(args.dropout_prob)(doc_output_1)
        doc_output_1 = doc_senti_dense(doc_output_1)
        doc_prob_1 = softmax(doc_output_1)

        if args.doc_domain_layers > 0:
            doc_output_2 = doc_domain_cnn(doc_output_2)
        att_2, _ = doc_domain_att(doc_output_2)
        doc_output_2 = WeightedSum()([doc_output_2, att_2])
        doc_output_2 = Dropout(args.dropout_prob)(doc_output_2)
        doc_prob_2 = doc_domain_dense(doc_output_2)

        doc_model = Model(inputs=[doc_input_1, doc_input_2], outputs=[doc_prob_1, doc_prob_2])
       
    else:
        doc_model = None


    ####################################################
    # initialize word embeddings
    ####################################################

    logger.info('Initializing lookup table')


    # Load pre-trained word vectors.
    # To save the loading time, here we load from the extracted subsets of the original embeddings, 
    # which only contains the embeddings of words in the vocab. 
    if args.use_doc:
        emb_path_gen = '../glove/%s_.txt'%(args.domain)
        emb_path_domain = '../domain_specific_emb/%s_.txt'%(args.domain)
    else:
        emb_path_gen = '../glove/%s.txt'%(args.domain)
        emb_path_domain = '../domain_specific_emb/%s.txt'%(args.domain)




    # Load pre-trained word vectors from the orginal large files
    # If you are loading from ssd, the process would only take 1-2 mins
    # If you are loading from hhd, the process would take a few hours at first try, 
    # and would take 1-2 mins in subsequent repeating runs (due to cache performance). 

    # emb_path_gen = '../glove.840B.300d.txt'
    # if args.domain == 'lt':
    #     emb_path_domain = '../laptop_emb.vec'
    # else:
    #     emb_path_domain = '../restaurant_emb.vec'



    aspect_model.get_layer('word_emb').set_weights(init_emb(aspect_model.get_layer('word_emb').get_weights(), vocab, emb_path_gen, emb_path_domain))

    logger.info('  Done')
    
    return aspect_model, doc_model



