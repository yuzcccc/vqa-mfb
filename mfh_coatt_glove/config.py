#training parameters
TRAIN_GPU_ID = 0
TEST_GPU_ID = 0
BATCH_SIZE = 64
VAL_BATCH_SIZE = 32
PRINT_INTERVAL = 100
VALIDATE_INTERVAL = 10000 
MAX_ITERATIONS = 100000
RESTORE_ITER = 0 # iteration to restore. *.solverstate file is needed!
# what data to use for training
TRAIN_DATA_SPLITS = 'train+val'
# what data to use for the vocabulary
QUESTION_VOCAB_SPACE = 'train+val'
ANSWER_VOCAB_SPACE = 'train+val' # test/test-dev/genome should not appear here

#network parameters
NUM_OUTPUT_UNITS = 3000 # This is the answer vocabulary size
MFB_FACTOR_NUM = 5
MFB_OUT_DIM = 1000
LSTM_UNIT_NUM = 1024
JOINT_EMB_SIZE = MFB_FACTOR_NUM*MFB_OUT_DIM
NUM_IMG_GLIMPSE = 2
NUM_QUESTION_GLIMPSE = 2
IMG_FEAT_WIDTH = 14
IMG_FEAT_SIZE = IMG_FEAT_WIDTH * IMG_FEAT_WIDTH
MAX_WORDS_IN_QUESTION = 15
LSTM_DROPOUT_RATIO = 0.3
MFB_DROPOUT_RATIO = 0.1

# vqa tools - get from https://github.com/VT-vision-lab/VQA
VQA_TOOLS_PATH = '/home/yuz/data/VQA/PythonHelperTools'
VQA_EVAL_TOOLS_PATH = '/home/yuz/data/VQA/PythonEvaluationTools'

# location of the data
VQA_PREFIX = '/home/yuz/data/VQA'

feat = 'res5c'
DATA_PATHS = {
	'train': {
		'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_train2014_questions.json',
		'ans_file': VQA_PREFIX + '/Annotations/mscoco_train2014_annotations.json',
		'features_prefix': VQA_PREFIX + '/Features/ms_coco/resnet_%s_bgrms_large/train2014/COCO_train2014_'%feat
	},
	'val': {
		'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_val2014_questions.json',
		'ans_file': VQA_PREFIX + '/Annotations/mscoco_val2014_annotations.json',
		'features_prefix': VQA_PREFIX + '/Features/ms_coco/resnet_%s_bgrms_large/val2014/COCO_val2014_'%feat
	},
	'test-dev': {
		'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_test-dev2015_questions.json',
		'features_prefix': VQA_PREFIX + '/Features/ms_coco/resnet_%s_bgrms_large/test2015/COCO_test2015_'%feat
	},
	'test': {
		'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_test2015_questions.json',
		'features_prefix': VQA_PREFIX + '/Features/ms_coco/resnet_%s_bgrms_large/test2015/COCO_test2015_'%feat
	},
	'genome': {
		'genome_file': VQA_PREFIX + '/Questions/OpenEnded_genome_train_questions.json',
		'features_prefix': VQA_PREFIX + '/Features/genome/feat_resnet-152/resnet_%s_bgrms_large/'%feat
	}
}
