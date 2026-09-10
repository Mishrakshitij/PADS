def get_loss_action_state_from_model(initial_state, contextFeature_placeholder,
                                 bowVec_placeholder, lastAction_placeholder, labels_placeholder, mask=False,
                                 bit_vec_placeholder=None, W_embedding=None, questionID_placeholder=None,
                                     sent_embedding_placeholder=None, newFeature_placeholder=None, transition_placeholder=None):
    """
    return loss, action for current turn, state
    :param step_i:
    :param initial_state:
    :param W_embedding:
    :param questionID:
    :param contextFeature_placeholder:
    :param bowVec_placeholder:
    :param lastAction_placeholder:
    :param labels_placeholder:
    :return:
    """
    # 4. word embedding vec:
    if mask and bit_vec_placeholder is None:
        raise Exception('mask is on, so please provide the bit vector')
    if FLAGS.sent_embedding == "old":
        wordEmbeddings = tf.nn.embedding_lookup(W_embedding, questionID_placeholder)
        wordEmbeddings_mean = tf.reduce_mean(wordEmbeddings, axis=0)
        wordEmbeddings_mean = tf.reshape(wordEmbeddings_mean, [-1])
        if FLAGS.new_feature == "new_only":
            print("using new feature only\n\n\n\n\n\n\n\n\n\n")
            vec_concat = tf.concat([contextFeature_placeholder, wordEmbeddings_mean, bowVec_placeholder, lastAction_placeholder, newFeature_placeholder], 0)
        elif FLAGS.new_feature == "transition_only":
            vec_concat = tf.concat([contextFeature_placeholder, wordEmbeddings_mean, bowVec_placeholder, lastAction_placeholder, transition_placeholder], 0)
        elif FLAGS.new_feature == "new_and_transition":
            vec_concat = tf.concat([contextFeature_placeholder, wordEmbeddings_mean, bowVec_placeholder, lastAction_placeholder, newFeature_placeholder, transition_placeholder], 0)
        else:
            vec_concat = tf.concat([contextFeature_placeholder, wordEmbeddings_mean, bowVec_placeholder, lastAction_placeholder], 0)
    elif FLAGS.sent_embedding == 'new':
        print("using SIF\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        if FLAGS.new_feature == "new":
            vec_concat = tf.concat(
                [contextFeature_placeholder, sent_embedding_placeholder, bowVec_placeholder, lastAction_placeholder, newFeature_placeholder], 0)
        else:
            vec_concat = tf.concat(
                [contextFeature_placeholder, sent_embedding_placeholder, bowVec_placeholder, lastAction_placeholder, transition_placeholder], 0)
    vec_concat = tf.expand_dims(vec_concat, axis=0)

    with tf.variable_scope('HCN') as scope:
        rnn_cell = tf.contrib.rnn.LSTMCell(HIDDEN_DIM, state_is_tuple=True)
        # if step_i == 0:
        #     initial_state = rnn_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)
        # elif step_i > 0:
        #     scope.reuse_variables()
        rnn_output, rnn_state = rnn_cell(vec_concat, initial_state)

        # retain the state for next turn
        initial_state = rnn_state

        with tf.variable_scope('Dense') as scope_dense:
            dense_output = tf.layers.dense(rnn_output, OUTPUT_DIM, activation=None)

    if mask:
        print("in mask")
        masked_output = dense_output * bit_vec_placeholder
        """
        masked_output = tf.multiply(softmax_dense_output, bit_vec_placeholder)
        # normalize
        denominator = tf.reduce_sum(masked_output)
        normalized_output = tf.divide(masked_output, denominator)
        loss = tf.reduce_mean(-tf.reduce_sum(labels_placeholder * tf.log(normalized_output), reduction_indices=[1]))
        action_id = tf.argmax(normalized_output, axis=1)
        """
        loss = tf.losses.softmax_cross_entropy(labels_placeholder, masked_output)
        action_id = tf.argmax(tf.nn.softmax(masked_output), axis=1)
    else:
        print("dense")
        #dense_output = tf.Print(dense_output, [dense_output], message='dense', summarize=30, first_n=500)
        print("softmax_dense")
        softmax_dense_output = tf.nn.softmax(dense_output)
        #softmax_dense_output = tf.Print(softmax_dense_output, [softmax_dense_output], message='softmax_dense', summarize=30, first_n=500)
        loss = tf.losses.softmax_cross_entropy(labels_placeholder, dense_output)
        action_id = tf.argmax(softmax_dense_output, axis=1)

    return loss, action_id, initial_state
