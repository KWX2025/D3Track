

from rgbt.utils import RGBT_start
import rgbt     
from seqList import *
RGBT_start()
gtot = rgbt.GTOT()


result_path="your tracking result path"   
seq_list,length = where_seq_already(result_path, prefix="")
print("seq num: ", length)




gtot(
    tracker_name="tracker_name1",
    result_path=result_path,
    seqs=seq_list
)






if __name__=="__main__":

    mpr_dict = gtot.MPR(seqs=seq_list)

    print('')
    for k,v in mpr_dict.items():
        print(k, "PR", round(v[0]*100, 1))

    print('')
    msr_dict = gtot.MSR(seqs=seq_list)

    for k,v in msr_dict.items():
        print(k, "SR", round(v[0]*100, 1))
