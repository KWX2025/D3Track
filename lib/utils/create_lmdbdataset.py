import lmdb
import os
import sys
prj_path = os.getcwd()
if prj_path not in sys.path:
    sys.path.append(prj_path)
from lib.train.dataset.rgbt234 import RGBT234
from lib.train.dataset.rgbt210 import RGBT210
from lib.train.dataset.gtot import GTOT
from lib.train.dataset.LasHeR_trainingSet import LasHeR_trainingSet
from lib.train.dataset.LasHeR_testingSet import LasHeR_testingSet




class LmdbDataset:
    def __init__(self, lmdb_path, map_size=None) -> None:
        
        if map_size==None:
            self._lmdb_env = lmdb.open(lmdb_path)
        else:
            self._lmdb_env = lmdb.open(lmdb_path, map_size=map_size)

    def is_exist(self, key) -> bool:
        key = str(key).encode()
        res = False if self._lmdb_txn.get(key) is None else True
        return res

    def begin(self):
        self._lmdb_txn = self._lmdb_env.begin(write=True)

    def commit(self):
        self._lmdb_txn.commit()

    def end(self):
        self._lmdb_env.close()

    def img2bin(self, img_path):
        
        with open(img_path, 'rb') as f:
            image_bin = f.read()
        return image_bin

    def lbl2bin(self, lbl_path):
        with open(lbl_path, 'rb') as f:
            lbl_bin = f.read()
        return lbl_bin

    def write_img(self, key, img_path, ):
        key = str(key)
        v = self.img2bin(img_path)
        self._lmdb_txn.put(key.encode(), v)

    def write_label(self, key, lbl_path):
        key = str(key)
        v = self.lbl2bin(lbl_path)
        self._lmdb_txn.put(key.encode(), v)




if __name__=="__main__":

    dataset = [
        
        LasHeR_trainingSet(''),
        LasHeR_testingSet('')]

    
    dataset_path = ""
    map_size = 1024*1024*1024*320       
    lds = LmdbDataset(dataset_path, map_size)
    

    
    for datas in dataset:
        dataset_name = datas.get_name().lower()

        
        for seq_id in range(len(datas.sequence_list)):
            lds.begin()
            seq_name = datas.sequence_list[seq_id]
            seq_path = os.path.join(datas.root, seq_name)
            key = dataset_name+'.'+seq_name

            t_key = key+'.infrared_lbl'
            if lds.is_exist(t_key):
                lds.commit()
                print(dataset_name, seq_id, seq_name, '---------skip')
                continue
            else:
                print(dataset_name, seq_id, seq_name)

            
            frame_path_v=[]
            for item in sorted([p for p in os.listdir(os.path.join(seq_path, 'visible')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']]):
                frame_path_v.append( os.path.join(seq_path, 'visible', item) )
            
            frame_path_i=[]
            for item in sorted([p for p in os.listdir(os.path.join(seq_path, 'infrared')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']]):
                frame_path_i.append( os.path.join(seq_path, 'infrared', item) )

    
            for i,frame_path in enumerate(frame_path_i):
                t_key = key+'.infrared.'+str(i)
                if not lds.is_exist(t_key):
                    lds.write_img(t_key, frame_path)

            for i,frame_path in enumerate(frame_path_v):
                t_key = key+'.visible.'+str(i)
                if not lds.is_exist(t_key):
                    lds.write_img(t_key, frame_path)
            
            if not lds.is_exist(key+'.init_lbl'):
                lds.write_label(key+'.init_lbl', os.path.join(seq_path, 'init.txt'))
            if not lds.is_exist(key+'.visible_lbl'):
                lds.write_label(key+'.visible_lbl', os.path.join(seq_path, 'visible.txt'))
            if not lds.is_exist(key+'.infrared_lbl'):
                lds.write_label(key+'.infrared_lbl', os.path.join(seq_path, 'infrared.txt'))
            lds.commit()

    
    lds.end()
