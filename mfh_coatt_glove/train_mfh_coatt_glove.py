import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

from vqa_data_layer_kld import VQADataProvider
from utils import exec_validation, drawgraph
import config
import time

def get_solver(folder):
    s = caffe_pb2.SolverParameter()
    s.train_net = './%s/proto_train.prototxt'%folder
    s.snapshot = 10000
    s.snapshot_prefix = './%s/'%folder
    s.max_iter = int(config.MAX_ITERATIONS)
    s.display = int(config.VALIDATE_INTERVAL)
    s.type = 'Adam'
    s.stepsize = int(config.MAX_ITERATIONS*0.4)
    s.gamma = 0.25
    s.lr_policy = "step"
    s.base_lr = 0.0007
    s.momentum = 0.9
    s.momentum2 = 0.999
    s.weight_decay = 0.000
    s.clip_gradients = 10
    return s

def get_auxiliary_json():
    aux = {}
    aux["batch_size"] = int(config.VAL_BATCH_SIZE)
    aux["data_shape"] = [2048,14,14]
    aux["img_feature_prefix"] = config.DATA_PATHS['test']['features_prefix']
    aux["glove"] = True
    return aux


def mfb_coatt(mode, batchsize, T, question_vocab_size, folder):
    n = caffe.NetSpec()
    mode_str = json.dumps({'mode':mode, 'batchsize':batchsize,'folder':folder})
    if mode == 'val':
        n.data, n.cont, n.img_feature, n.label, n.glove = L.Python( \
            module='vqa_data_layer', layer='VQADataProviderLayer', \
            param_str=mode_str, ntop=5 )
    else:
        n.data, n.cont, n.img_feature, n.label, n.glove = L.Python(\
            module='vqa_data_layer_kld', layer='VQADataProviderLayer', \
            param_str=mode_str, ntop=5 ) 
    n.embed = L.Embed(n.data, input_dim=question_vocab_size, num_output=300, \
                         weight_filler=dict(type='xavier'))
    n.embed_tanh = L.TanH(n.embed) 
    concat_word_embed = [n.embed_tanh, n.glove]
    n.concat_embed = L.Concat(*concat_word_embed, concat_param={'axis': 2}) # T x N x 600

    # LSTM
    n.lstm1 = L.LSTM(\
                   n.concat_embed, n.cont,\
                   recurrent_param=dict(\
                       num_output=config.LSTM_UNIT_NUM,\
                       weight_filler=dict(type='xavier')))
    n.lstm1_droped = L.Dropout(n.lstm1,dropout_param={'dropout_ratio':config.LSTM_DROPOUT_RATIO})
    n.lstm1_resh = L.Permute(n.lstm1_droped, permute_param=dict(order=[1,2,0]))
    n.lstm1_resh2 = L.Reshape(n.lstm1_resh, \
            reshape_param=dict(shape=dict(dim=[0,0,0,1])))

    '''
    Question Attention
    '''
    n.qatt_conv1 = L.Convolution(n.lstm1_resh2, kernel_size=1, stride=1, num_output=512, pad=0,
                                           weight_filler=dict(type='xavier'))
    n.qatt_relu = L.ReLU(n.qatt_conv1)
    n.qatt_conv2 = L.Convolution(n.qatt_relu, kernel_size=1, stride=1, num_output=config.NUM_QUESTION_GLIMPSE, pad=0,
                                           weight_filler=dict(type='xavier')) 
    n.qatt_reshape = L.Reshape(n.qatt_conv2, reshape_param=dict(shape=dict(dim=[-1,config.NUM_QUESTION_GLIMPSE,config.MAX_WORDS_IN_QUESTION,1]))) # N*NUM_QUESTION_GLIMPSE*15
    n.qatt_softmax = L.Softmax(n.qatt_reshape, axis=2)

    qatt_maps = L.Slice(n.qatt_softmax,ntop=config.NUM_QUESTION_GLIMPSE,slice_param={'axis':1})
    dummy_lstm = L.DummyData(shape=dict(dim=[batchsize, 1]), data_filler=dict(type='constant', value=1), ntop=1)
    qatt_feature_list = []
    for i in xrange(config.NUM_QUESTION_GLIMPSE):
        if config.NUM_QUESTION_GLIMPSE == 1:
            n.__setattr__('qatt_feat%d'%i, L.SoftAttention(n.lstm1_resh2, qatt_maps, dummy_lstm))
        else:
            n.__setattr__('qatt_feat%d'%i, L.SoftAttention(n.lstm1_resh2, qatt_maps[i], dummy_lstm))    
        qatt_feature_list.append(n.__getattr__('qatt_feat%d'%i))
    n.qatt_feat_concat = L.Concat(*qatt_feature_list) 
    '''
    Image Attention with MFB
    '''
    n.q_feat_resh = L.Reshape(n.qatt_feat_concat,reshape_param=dict(shape=dict(dim=[0,-1,1,1])))
    n.i_feat_resh = L.Reshape(n.img_feature,reshape_param=dict(shape=dict(dim=[0,-1,config.IMG_FEAT_WIDTH,config.IMG_FEAT_WIDTH])))
    
    n.iatt_q_proj = L.InnerProduct(n.q_feat_resh, num_output = config.JOINT_EMB_SIZE, 
                                   weight_filler=dict(type='xavier'))
    n.iatt_q_resh = L.Reshape(n.iatt_q_proj, reshape_param=dict(shape=dict(dim=[-1,config.JOINT_EMB_SIZE,1,1])))  
    n.iatt_q_tile1 = L.Tile(n.iatt_q_resh, axis=2, tiles=config.IMG_FEAT_WIDTH)
    n.iatt_q_tile2 = L.Tile(n.iatt_q_tile1, axis=3, tiles=config.IMG_FEAT_WIDTH)


    n.iatt_i_conv = L.Convolution(n.i_feat_resh, kernel_size=1, stride=1, num_output=config.JOINT_EMB_SIZE, pad=0,
                                 weight_filler=dict(type='xavier')) 
    n.iatt_i_resh1 = L.Reshape(n.iatt_i_conv, reshape_param=dict(shape=dict(dim=[-1,config.JOINT_EMB_SIZE,
                                                                      config.IMG_FEAT_WIDTH,config.IMG_FEAT_WIDTH])))
    n.iatt_iq_eltwise = L.Eltwise(n.iatt_q_tile2, n.iatt_i_resh1, eltwise_param=dict(operation=0))
    n.iatt_iq_droped = L.Dropout(n.iatt_iq_eltwise, dropout_param={'dropout_ratio':config.MFB_DROPOUT_RATIO})
    n.iatt_iq_resh2 = L.Reshape(n.iatt_iq_droped, reshape_param=dict(shape=dict(dim=[-1,config.JOINT_EMB_SIZE,196,1])))
    n.iatt_iq_permute1 = L.Permute(n.iatt_iq_resh2, permute_param=dict(order=[0,2,1,3]))
    n.iatt_iq_resh2 = L.Reshape(n.iatt_iq_permute1, reshape_param=dict(shape=dict(dim=[-1,config.IMG_FEAT_SIZE,
                                                                       config.MFB_OUT_DIM,config.MFB_FACTOR_NUM])))
    n.iatt_iq_sumpool = L.Pooling(n.iatt_iq_resh2, pool=P.Pooling.SUM, \
                              pooling_param=dict(kernel_w=config.MFB_FACTOR_NUM, kernel_h=1))
    n.iatt_iq_permute2 = L.Permute(n.iatt_iq_sumpool, permute_param=dict(order=[0,2,1,3]))
    
    n.iatt_iq_sqrt = L.SignedSqrt(n.iatt_iq_permute2)
    n.iatt_iq_l2 = L.L2Normalize(n.iatt_iq_sqrt)


    ## 2 conv layers 1000 -> 512 -> 2
    n.iatt_conv1 = L.Convolution(n.iatt_iq_l2, kernel_size=1, stride=1, num_output=512, pad=0, 
                                weight_filler=dict(type='xavier'))
    n.iatt_relu = L.ReLU(n.iatt_conv1)
    n.iatt_conv2 = L.Convolution(n.iatt_relu, kernel_size=1, stride=1, num_output=config.NUM_IMG_GLIMPSE, pad=0,
                                           weight_filler=dict(type='xavier')) 
    n.iatt_resh = L.Reshape(n.iatt_conv2, reshape_param=dict(shape=dict(dim=[-1,config.NUM_IMG_GLIMPSE,config.IMG_FEAT_SIZE])))
    n.iatt_softmax = L.Softmax(n.iatt_resh, axis=2)
    n.iatt_softmax_resh = L.Reshape(n.iatt_softmax,reshape_param=dict(shape=dict(dim=[-1,config.NUM_IMG_GLIMPSE,config.IMG_FEAT_WIDTH,config.IMG_FEAT_WIDTH])))
    iatt_maps = L.Slice(n.iatt_softmax_resh, ntop=config.NUM_IMG_GLIMPSE,slice_param={'axis':1})
    dummy = L.DummyData(shape=dict(dim=[batchsize, 1]), data_filler=dict(type='constant', value=1), ntop=1)
    iatt_feature_list = []
    for i in xrange(config.NUM_IMG_GLIMPSE):
        if config.NUM_IMG_GLIMPSE == 1:
            n.__setattr__('iatt_feat%d'%i, L.SoftAttention(n.i_feat_resh, iatt_maps, dummy))
        else:
            n.__setattr__('iatt_feat%d'%i, L.SoftAttention(n.i_feat_resh, iatt_maps[i], dummy))
        n.__setattr__('iatt_feat%d_resh'%i, L.Reshape(n.__getattr__('iatt_feat%d'%i), \
                                reshape_param=dict(shape=dict(dim=[0,-1]))))
        iatt_feature_list.append(n.__getattr__('iatt_feat%d_resh'%i))
    n.iatt_feat_concat = L.Concat(*iatt_feature_list)
    n.iatt_feat_concat_resh = L.Reshape(n.iatt_feat_concat, reshape_param=dict(shape=dict(dim=[0,-1,1,1])))
    
    '''
    Fine-grained Image-Question MFH fusion
    '''
    n.mfb_q_o2_proj = L.InnerProduct(n.q_feat_resh, num_output=config.JOINT_EMB_SIZE,
                                  weight_filler=dict(type='xavier'))
    n.mfb_i_o2_proj = L.InnerProduct(n.iatt_feat_concat_resh, num_output=config.JOINT_EMB_SIZE,
                                  weight_filler=dict(type='xavier'))
    n.mfb_iq_o2_eltwise = L.Eltwise(n.mfb_q_o2_proj, n.mfb_i_o2_proj, eltwise_param=dict(operation=0))
    n.mfb_iq_o2_drop = L.Dropout(n.mfb_iq_o2_eltwise, dropout_param={'dropout_ratio':config.MFB_DROPOUT_RATIO})
    n.mfb_iq_o2_resh = L.Reshape(n.mfb_iq_o2_drop, reshape_param=dict(shape=dict(dim=[-1,1,config.MFB_OUT_DIM,config.MFB_FACTOR_NUM])))
    n.mfb_iq_o2_sumpool = L.Pooling(n.mfb_iq_o2_resh, pool=P.Pooling.SUM, \
                                      pooling_param=dict(kernel_w=config.MFB_FACTOR_NUM, kernel_h=1))
    n.mfb_o2_out = L.Reshape(n.mfb_iq_o2_sumpool,\
                                    reshape_param=dict(shape=dict(dim=[-1,config.MFB_OUT_DIM])))
    n.mfb_o2_sign_sqrt = L.SignedSqrt(n.mfb_o2_out)
    n.mfb_o2_l2 = L.L2Normalize(n.mfb_o2_sign_sqrt)

    n.mfb_q_o3_proj = L.InnerProduct(n.q_feat_resh, num_output=config.JOINT_EMB_SIZE,
                                  weight_filler=dict(type='xavier'))
    n.mfb_i_o3_proj = L.InnerProduct(n.iatt_feat_concat_resh, num_output=config.JOINT_EMB_SIZE,
                                  weight_filler=dict(type='xavier'))
    n.mfb_iq_o3_eltwise = L.Eltwise(n.mfb_q_o3_proj, n.mfb_i_o3_proj,n.mfb_iq_o2_drop, eltwise_param=dict(operation=0))
    n.mfb_iq_o3_drop = L.Dropout(n.mfb_iq_o3_eltwise, dropout_param={'dropout_ratio':config.MFB_DROPOUT_RATIO})
    n.mfb_iq_o3_resh = L.Reshape(n.mfb_iq_o3_drop, reshape_param=dict(shape=dict(dim=[-1,1,config.MFB_OUT_DIM,config.MFB_FACTOR_NUM])))
    n.mfb_iq_o3_sumpool = L.Pooling(n.mfb_iq_o3_resh, pool=P.Pooling.SUM, \
                                      pooling_param=dict(kernel_w=config.MFB_FACTOR_NUM, kernel_h=1))
    n.mfb_o3_out = L.Reshape(n.mfb_iq_o3_sumpool,\
                                    reshape_param=dict(shape=dict(dim=[-1,config.MFB_OUT_DIM])))
    n.mfb_o3_sign_sqrt = L.SignedSqrt(n.mfb_o3_out)
    n.mfb_o3_l2 = L.L2Normalize(n.mfb_o3_sign_sqrt)

    n.mfb_o23_l2 = L.Concat(n.mfb_o2_l2,n.mfb_o3_l2)

    n.prediction = L.InnerProduct(n.mfb_o23_l2, num_output=config.NUM_OUTPUT_UNITS,
                                  weight_filler=dict(type='xavier'))

    if mode == 'val':
        n.loss = L.SoftmaxWithLoss(n.prediction, n.label)
    else:
        n.loss = L.SoftmaxKLDLoss(n.prediction, n.label) 
    return n.to_proto()

def make_answer_vocab(adic, vocab_size):
    """
    Returns a dictionary that maps words to indices.
    """
    adict = {'':0}
    nadict = {'':1000000}
    vid = 1
    for qid in adic.keys():
        answer_obj = adic[qid]
        answer_list = [ans['answer'] for ans in answer_obj]
        
        for q_ans in answer_list:
            # create dict
            if adict.has_key(q_ans):
                nadict[q_ans] += 1
            else:
                nadict[q_ans] = 1
                adict[q_ans] = vid
                vid +=1

    # debug
    nalist = []
    for k,v in sorted(nadict.items(), key=lambda x:x[1]):
        nalist.append((k,v))

    # remove words that appear less than once 
    n_del_ans = 0
    n_valid_ans = 0
    adict_nid = {}
    for i, w in enumerate(nalist[:-vocab_size]):
        del adict[w[0]]
        n_del_ans += w[1]
    for i, w in enumerate(nalist[-vocab_size:]):
        n_valid_ans += w[1]
        adict_nid[w[0]] = i
    
    return adict_nid

def make_question_vocab(qdic):
    """
    Returns a dictionary that maps words to indices.
    """
    vdict = {'':0}
    vid = 1
    for qid in qdic.keys():
        # sequence to list
        q_str = qdic[qid]['qstr']
        q_list = VQADataProvider.seq_to_list(q_str)

        # create dict
        for w in q_list:
            if not vdict.has_key(w):
                vdict[w] = vid
                vid +=1

    return vdict

def make_vocab_files():
    """
    Produce the question and answer vocabulary files.
    """
    print 'making question vocab...', config.QUESTION_VOCAB_SPACE
    qdic, _ = VQADataProvider.load_data(config.QUESTION_VOCAB_SPACE)
    question_vocab = make_question_vocab(qdic)
    print 'making answer vocab...', config.ANSWER_VOCAB_SPACE
    _, adic = VQADataProvider.load_data(config.ANSWER_VOCAB_SPACE)
    answer_vocab = make_answer_vocab(adic, config.NUM_OUTPUT_UNITS)
    return question_vocab, answer_vocab

def main():
    folder = 'mfh_coatt_glove_q%dv%d_%s'%(config.NUM_QUESTION_GLIMPSE, config.NUM_IMG_GLIMPSE,config.TRAIN_DATA_SPLITS)
    if not os.path.exists('./%s'%folder):
        os.makedirs('./%s'%folder)

    question_vocab, answer_vocab = {}, {}
    if os.path.exists('./%s/vdict.json'%folder) and os.path.exists('./%s/adict.json'%folder):
        print 'restoring vocab'
        with open('./%s/vdict.json'%folder,'r') as f:
            question_vocab = json.load(f)
        with open('./%s/adict.json'%folder,'r') as f:
            answer_vocab = json.load(f)
    else:
        question_vocab, answer_vocab = make_vocab_files()
        with open('./%s/vdict.json'%folder,'w') as f:
            json.dump(question_vocab, f)
        with open('./%s/adict.json'%folder,'w') as f:
            json.dump(answer_vocab, f)

    print 'question vocab size:', len(question_vocab)
    print 'answer vocab size:', len(answer_vocab)

    with open('./%s/proto_train.prototxt'%folder, 'w') as f:
        f.write(str(mfb_coatt(config.TRAIN_DATA_SPLITS, config.BATCH_SIZE, \
            config.MAX_WORDS_IN_QUESTION, len(question_vocab), folder)))
    
    with open('./%s/proto_test.prototxt'%folder, 'w') as f:
        f.write(str(mfb_coatt('val', config.VAL_BATCH_SIZE, \
            config.MAX_WORDS_IN_QUESTION, len(question_vocab), folder)))

    with open('./%s/solver.prototxt'%folder, 'w') as f:
        f.write(str(get_solver(folder)))    
    with open('./%s/auxiliary.json'%folder, 'w') as f:
        json.dump(get_auxiliary_json(),f, indent=2)

    caffe.set_device(config.TRAIN_GPU_ID)
    caffe.set_mode_gpu()
    solver = caffe.get_solver('./%s/solver.prototxt'%folder)

    train_loss = np.zeros(config.MAX_ITERATIONS+1)
    results = []

    if config.RESTORE_ITER:
        restore_iter = config.RESTORE_ITER
        solver.restore('./%s/_iter_%d.solverstate'%(folder,restore_iter))
    else:
        restore_iter = 0
    
    start = time.clock()
    for it in range(restore_iter, config.MAX_ITERATIONS+1):
        solver.step(1)
    
        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data
   
        if it % config.PRINT_INTERVAL == 0 and it != 0:
            elapsed = (time.clock() - start)
            print 'Iteration:', it
            c_mean_loss = train_loss[it-config.PRINT_INTERVAL:it].mean()
            print 'Train loss:', c_mean_loss, ' Elapsed seconds:', elapsed
            start = time.clock()
        if it % config.VALIDATE_INTERVAL == 0 and it != restore_iter:
            model_name = './%s/tmp.caffemodel'%(folder)
            solver.net.save(model_name)
            print 'Validating...'
            
            # for test-dev /test set. the json file will be generated under the <folder> file
            exec_validation(config.TEST_GPU_ID, 'test-dev', model_name, it=it, folder=folder)
            caffe.set_device(config.TRAIN_GPU_ID)
            ''' 
            #for val set. the accuracy will be computed and ploted 
            test_loss, acc_overall, acc_per_ques, acc_per_ans = exec_validation(config.TEST_GPU_ID, 'val', model_name, it=it, folder=folder)
            caffe.set_device(config.TRAIN_GPU_ID)
            print 'Test loss:', test_loss
            print 'Accuracy:', acc_overall
            print 'Test per ans', acc_per_ans
            results.append([it, c_mean_loss, test_loss, acc_overall, acc_per_ques, acc_per_ans])
            best_result_idx = np.array([x[3] for x in results]).argmax()
            print 'Best accuracy of', results[best_result_idx][3], 'was at iteration', results[best_result_idx][0]
            drawgraph(results,folder,config.MFB_FACTOR_NUM,config.MFB_OUT_DIM,prefix='mfh_coatt_glove')
            ''' 
if __name__ == '__main__':
    main()
