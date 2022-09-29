import pickle
import os


# save txt
save_dir = './'
# get time




pickle_data_path = '../reid/reid-matching/tools/test_cluster.pkl'
f = open(pickle_data_path,'rb')   #pickle_data_path为.pickle文件的路径；
info = pickle.load(f)
print(type(info))


# write txt
fs = open(pickle_data_path.split('/')[-1]+ '.txt', 'a')
fs.write(str(info))
fs.close()

f.close()  #别忘记close pickle文件
