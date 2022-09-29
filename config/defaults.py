from yacs.config import CfgNode as CN

_C = CN()

_C.CHALLENGE_DATA_DIR = ''
_C.DET_SOURCE_DIR = ''
_C.REID_MODEL = ''
_C.REID_BACKBONE = ''
_C.REID_SIZE_TEST = [256, 256]

_C.DET_IMG_DIR = ''
_C.DATA_DIR = ''
_C.ROI_DIR = ''
_C.CID_BIAS_DIR = ''
_C.LABEL_DIR = ''  # added by wgj

_C.USE_ST_FILTER = True  # get_sim_mat
_C.USE_CAMERA = True  # sub_clu
_C.USE_ROI = False # gen_res
_C.USE_RERANK = False  # get_sim_mat
_C.USE_FF = False  # get_sim_mat
_C.USE_ZONE = False  # traj_fusion
_C.SCORE_THR = 0.5

_C.MCMT_OUTPUT_TXT = ''
_C.MTMC_VIS_OUTPUT = ''
