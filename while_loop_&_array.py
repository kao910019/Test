# 我不知道為啥以下兩種方式，不用TensorArray跟用TensorArray的，
# 不用的話在更新參數時會出現形狀不吻合的問題，但是使用TensorArray之後就沒問題了......
# 明明兩種結果輸出的數據應該都是一樣的才對，
# def build_evaluator_transformer(self, hparams, memory, encoder_id, response_flag, dropout_rate, trainable=True):
    #     batch_size = tf.shape(encoder_id)[0]
    #     init_loop_decoder_logits = tf.fill([batch_size, 0, hparams.num_dimensions], 0.0)
    #     init_loop_decoder_id = tf.fill([batch_size, 1], self.cls_id)
    #     init_loop_decoder_length = tf.fill([batch_size], 0)
    #     stop_flag = tf.constant(False)
    #     steps = tf.constant(1)
    #     def decode_loop(stop_flag, steps, loop_decoder__logits, loop_decoder_id, loop_decoder_length):
    #         logits, ids, length = self.build_decoder(hparams, memory, encoder_id, loop_decoder_id, response_flag, dropout_rate, training=False, trainable=trainable)
    #         rebuild_ids = tf.concat([tf.fill([batch_size, 1], self.cls_id), ids], axis = 1)
    #         stop_flag = tf.logical_or(tf.greater_equal(steps, hparams.max_length), 
    #                                   tf.reduce_all(tf.reduce_any(tf.logical_or(tf.equal(ids, self.sep_id), tf.equal(ids, self.pad_id)), 1),0))
    #         return stop_flag, tf.add(steps, 1), logits, rebuild_ids, length
        
    #     _, _, decoder_logits, decoder_id, decoder_length \
    #       = tf.while_loop(lambda stop_flag, *_: tf.logical_not(stop_flag), decode_loop,
    #                       loop_vars=[stop_flag, steps, init_loop_decoder_logits, init_loop_decoder_id, init_loop_decoder_length],
    #                       shape_invariants=[stop_flag.get_shape(), steps.get_shape(), tf.TensorShape([None, None, None]), tf.TensorShape([None, None]), tf.TensorShape([None])],
    #                       back_prop = True, name='decode_loop')
          
    #     return decoder_logits, decoder_id[:,1:], decoder_length
    
    def build_evaluator_transformer(self, hparams, memory, encoder_id, response_flag, dropout_rate, trainable=True):
        batch_size = tf.shape(encoder_id)[0]
        init_loop_decoder_logits = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=1, dynamic_size=True, clear_after_read=False) 
        init_loop_decoder_id = tf.TensorArray(dtype=tf.int32, infer_shape=False, size=1, dynamic_size=True, clear_after_read=False) 
        init_loop_decoder_length = tf.TensorArray(dtype=tf.int32, infer_shape=False, size=1, dynamic_size=True, clear_after_read=False) 
        init_loop_decoder_logits = init_loop_decoder_logits.write(0, tf.fill([batch_size, 0, self.vocab_size], 0.0))
        init_loop_decoder_id = init_loop_decoder_id.write(0, tf.fill([batch_size, 0], 0))
        init_loop_decoder_length = init_loop_decoder_length.write(0, tf.fill([batch_size], 0))
        def decode_loop(stop_flag, step, loop_decoder_logits, loop_decoder_id, loop_decoder_length):
            input_id = tf.concat([tf.fill([batch_size, 1], self.cls_id), loop_decoder_id.read(step-1)], axis = 1)
            logits, ids, length = self.build_decoder(hparams, memory, encoder_id, input_id, response_flag, dropout_rate, training=False, trainable=trainable)
            
            loop_decoder_logits = loop_decoder_logits.write(step, logits)
            loop_decoder_id = loop_decoder_id.write(step, ids)
            loop_decoder_length = loop_decoder_length.write(step, length)
            
            stop_flag = tf.logical_or(tf.greater_equal(step, hparams.max_length), 
                                      tf.reduce_all(tf.reduce_any(tf.logical_or(tf.equal(ids, self.sep_id), tf.equal(ids, self.pad_id)), 1), 0))
            return stop_flag, tf.add(step, 1), loop_decoder_logits, loop_decoder_id, loop_decoder_length
        _, step, decoder_logits, decoder_id, decoder_length \
          = tf.while_loop(lambda stop_flag, *_: tf.logical_not(stop_flag), decode_loop,
                          loop_vars=[False, 1, init_loop_decoder_logits, init_loop_decoder_id, init_loop_decoder_length],
                          back_prop = True, name='decode_loop')
        decoder_logits = tf.reshape(decoder_logits.read(step-1), [batch_size, -1, self.vocab_size])
        decoder_id = tf.reshape(decoder_id.read(step-1), [batch_size, -1])
        decoder_length = tf.reshape(decoder_length.read(step-1), [batch_size])
        return decoder_logits, decoder_id, decoder_length
