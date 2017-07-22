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
    s.snapshot = int(config.VALIDATE_INTERVAL)
    s.snapshot_prefix = './%s/'%folder
    s.max_iter = int(config.MAX_ITERATIONS)
    s.display = int(config.VALIDATE_INTERVAL)
    s.type = 'Adam'
    s.stepsize = int(config.MAX_ITERATIONS*0.4)
    s.gamma = 0.5
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
    aux["data_shape"] = [2048]
    aux["img_feature_prefix"] = config.DATA_PATHS['test']['features_prefix']
    aux["glove"] = False
    return aux


def mfb_baseline(mode, batchsize, T, question_vocab_size, folder):
    n = caffe.NetSpec()
    mode_str = json.dumps({'mode':mode, 'batchsize':batchsize,'folder':folder})
    if mode == 'val':
        n.data, n.cont, n.img_feature, n.label = L.Python( \
            module='vqa_data_layer', layer='VQADataProviderLayer', \
            param_str=mode_str, ntop=4 )
    else:
        n.data, n.cont, n.img_feature, n.label = L.Python(\
            module='vqa_data_layer_kld', layer='VQADataProviderLayer', \
            param_str=mode_str, ntop=4 ) 
    n.embed = L.Embed(n.data, input_dim=question_vocab_size, num_output=300, \
                         weight_filler=dict(type='xavier'))
    n.embed_tanh = L.TanH(n.embed) 

    # LSTM
    n.lstm1 = L.LSTM(\
                   n.embed_tanh, n.cont,\
                   recurrent_param=dict(\
                       num_output=config.LSTM_UNIT_NUM,\
                       weight_filler=dict(type='xavier')))
    tops1 = L.Slice(n.lstm1, ntop=config.MAX_WORDS_IN_QUESTION, slice_param={'axis':0})
    for i in xrange(config.MAX_WORDS_IN_QUESTION-1):
        n.__setattr__('slice_first'+str(i), tops1[int(i)])
        n.__setattr__('silence_data_first'+str(i), L.Silence(tops1[int(i)],ntop=0))
    n.lstm1_out = tops1[config.MAX_WORDS_IN_QUESTION-1]
    n.lstm1_reshaped = L.Reshape(n.lstm1_out,\
                          reshape_param=dict(\
                              shape=dict(dim=[-1,1024])))
    n.q_feat = L.Dropout(n.lstm1_reshaped,dropout_param={'dropout_ratio':config.LSTM_DROPOUT_RATIO})
    '''
    Coarse Image-Question MFB fusion
    '''

    n.mfb_q_proj = L.InnerProduct(n.q_feat, num_output=config.JOINT_EMB_SIZE, 
                                  weight_filler=dict(type='xavier'))
    n.mfb_i_proj = L.InnerProduct(n.img_feature, num_output=config.JOINT_EMB_SIZE, 
                                  weight_filler=dict(type='xavier'))
    n.mfb_iq_eltwise = L.Eltwise(n.mfb_q_proj, n.mfb_i_proj, eltwise_param=dict(operation=0))
    n.mfb_iq_drop = L.Dropout(n.mfb_iq_eltwise, dropout_param={'dropout_ratio':config.MFB_DROPOUT_RATIO})
    n.mfb_iq_resh = L.Reshape(n.mfb_iq_drop, reshape_param=dict(shape=dict(dim=[-1,1,config.MFB_OUT_DIM,config.MFB_FACTOR_NUM])))
    n.mfb_iq_sumpool = L.Pooling(n.mfb_iq_resh, pool=P.Pooling.SUM, \
                                      pooling_param=dict(kernel_w=config.MFB_FACTOR_NUM, kernel_h=1))
    n.mfb_out = L.Reshape(n.mfb_iq_sumpool,\
                                    reshape_param=dict(shape=dict(dim=[-1,config.MFB_OUT_DIM])))
    n.mfb_sign_sqrt = L.SignedSqrt(n.mfb_out)
    n.mfb_l2 = L.L2Normalize(n.mfb_sign_sqrt) 
    
    n.prediction = L.InnerProduct(n.mfb_l2, num_output=config.NUM_OUTPUT_UNITS,
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
    folder = 'mfb_baseline_%s'%(config.TRAIN_DATA_SPLITS)
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
        f.write(str(mfb_baseline(config.TRAIN_DATA_SPLITS, config.BATCH_SIZE, \
            config.MAX_WORDS_IN_QUESTION, len(question_vocab), folder)))
    
    with open('./%s/proto_test.prototxt'%folder, 'w') as f:
        f.write(str(mfb_baseline('val', config.VAL_BATCH_SIZE, \
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
            ''' 
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
            drawgraph(results,folder,config.MFB_FACTOR_NUM,config.MFB_OUT_DIM,prefix='mfb_baseline')
             
if __name__ == '__main__':
    main()
