from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  

dataset_dict = dict(
    
    gtot=DatasetInfo(module=pt % "gtot", class_name="GTOTDataset", kwargs=dict()),
   
    rgbt210=DatasetInfo(module=pt % "rgbt210", class_name="RGBT210Dataset", kwargs=dict()),
    rgbt234=DatasetInfo(module=pt % "rgbt234", class_name="RGBT234Dataset", kwargs=dict()),
    rgbt234_lmdb=DatasetInfo(module=pt % "rgbt234lmdb", class_name="RGBT234LmdbDataset", kwargs=dict()),
    rgbt234_lmdb_tc=DatasetInfo(module=pt % "rgbt234lmdb", class_name="RGBT234LmdbDataset", kwargs={'attr':'TC'}),
    rgbt234_lmdb_li=DatasetInfo(module=pt % "rgbt234lmdb", class_name="RGBT234LmdbDataset", kwargs={'attr':'LI'}),
    rgbt234_lmdb_po=DatasetInfo(module=pt % "rgbt234lmdb", class_name="RGBT234LmdbDataset", kwargs={'attr':'PO'}),
    rgbt234_lmdb_no=DatasetInfo(module=pt % "rgbt234lmdb", class_name="RGBT234LmdbDataset", kwargs={'attr':'NO'}),
    rgbt234_lmdb_spec=DatasetInfo(module=pt % "rgbt234lmdb", class_name="RGBT234LmdbDataset", kwargs={'attr':'special'}),
    lasher=DatasetInfo(module=pt % "lasher", class_name="LasHeRDataset", kwargs=dict()),
    lashertestingset=DatasetInfo(module=pt % "lashertestingset", class_name="LasHeRtestingSetDataset", kwargs=dict()),
    lashertestingset_lmdb=DatasetInfo(module=pt % "lashertestinglmdb", class_name="LasHeRTestLmdbDataset", kwargs=dict()),
    vtuav=DatasetInfo(module=pt % "vtuav", class_name="VTUAVDataset", kwargs=dict()),
    

    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
)


def load_dataset(name: str):
    
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  
    return dataset.get_sequence_list()


def get_dataset(*args):
    
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset