# author: Gaojian Wang

import pandas as pd
import os

mot_label_dir = './test_labels/S04/'
out_mtmct_label_dir = './eval/'
if not os.path.exists(out_mtmct_label_dir):
    os.makedirs(out_mtmct_label_dir)

mot_label = {}
for root, dirs, files in os.walk(mot_label_dir):
    files = sorted(files)
    for file in files:
        mot_df = pd.read_csv(os.path.join(root, file), header=None)
        mot_df.columns = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Xworld', 'Yworld', 'Unuse', 'Unuse', 'Unuse']
        cam_id = int(file[1:4])
        print('cam_id:', cam_id)
        mot_df.insert(loc=0, column='CameraId', value=[cam_id]*len(mot_df))
        mot_df = mot_df.drop(mot_df.columns[[-1, -2, -3]], axis=1)
        mot_df = pd.DataFrame(mot_df, columns=['CameraId', 'Id', 'FrameId', 'X', 'Y', 'Width', 'Height', 'Xworld', 'Yworld'])
        mot_label[cam_id] = mot_df

mot_labels = []
for k,v in mot_label.items():
    mot_labels.append(v)

mtmct_label = pd.concat(mot_labels)
mtmct_label.to_csv(out_mtmct_label_dir + 'test.txt', sep=' ',index=False, header = False)


df = pd.read_csv(out_mtmct_label_dir + 'test.txt', header=None, sep=' ')

indexNames = df[df[3] < 0].index
for l in indexNames:
    df.loc[l, 5] = df.loc[l, 5] + df.loc[l, 3]
    df.loc[l, 3] = 0
    # print(df)

indexNamesy = df[df[4] < 0].index
for l in indexNamesy:
    df.loc[l, 6] = df.loc[l, 6] + df.loc[l, 4]
    df.loc[l, 4] = 0

os.remove(out_mtmct_label_dir + 'test.txt')
df.to_csv(out_mtmct_label_dir + 'test_gt_S04.txt', sep=' ', index=False, header=False)
