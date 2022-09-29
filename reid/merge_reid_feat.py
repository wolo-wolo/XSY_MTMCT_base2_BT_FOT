"""Merge reid feature from different models."""

import os
import pickle
import sys

from sklearn import preprocessing
import numpy as np
sys.path.append('../')
from config import cfg


def merge_feat(_cfg):
    """Save feature."""

    # NOTE: modify the ensemble list here
    ensemble_list = ['detect_reid1', 'detect_reid2', 'detect_reid3']
    all_feat_dir = _cfg.DATA_DIR.split('detect')[0]
    # /home/ubuntu/wgj/AIC21-MTMC/AIC21-MTMC-HW/datasets/detect

    # for cam in ['c001', 'c002', 'c003', 'c004', 'c005', 'c006']:
    for cam in ['c001', 'c002', 'c003']:
        feat_dic_list = []
        for feat_mode in ensemble_list:
            feat_pkl_file = os.path.join(all_feat_dir, feat_mode, cam,
                                         f'{cam}_dets_feat.pkl')
            # /home/ubuntu/wgj/AIC21-MTMC/AIC21-MTMC-HW/datasets/, detect_reid1, c00x
            feat_mode_dic = pickle.load(open(feat_pkl_file, 'rb'))
            feat_dic_list.append(feat_mode_dic)
        merged_dic = feat_dic_list[0].copy()
        for patch_name in merged_dic:
            patch_feature_list = []
            for feat_mode_dic in feat_dic_list:
                patch_feature_list.append(feat_mode_dic[patch_name]['feat'])
            patch_feature_array = np.array(patch_feature_list)
            patch_feature_array = preprocessing.normalize(patch_feature_array,
                                                          norm='l2', axis=1)
            patch_feature_mean = np.mean(patch_feature_array, axis=0)
            merged_dic[patch_name]['feat'] = patch_feature_mean

        merge_dir = os.path.join(all_feat_dir, 'detect_merge', cam)
        # datasets/detect_merge/c00x/
        if not os.path.exists(merge_dir):
            os.makedirs(merge_dir)

        merged_pkl_file = os.path.join(merge_dir, f'{cam}_dets_feat.pkl')
        pickle.dump(merged_dic, open(merged_pkl_file, 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        print('save pickle in %s' % merged_pkl_file)


def main():
    """Main method."""

    cfg.merge_from_file(f'../config/{sys.argv[1]}')
    cfg.freeze()
    merge_feat(cfg)


if __name__ == "__main__":
    main()
