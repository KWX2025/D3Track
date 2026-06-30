

import rgbt     
from seqList import *
rgbt234 = rgbt.RGBT234()

result_path= ""  
seq_list,length = where_seq_already(result_path, prefix="")
print("seq num: ", length)



rgbt234(
    tracker_name="tracker_name1",
    result_path=result_path,
    seqs=seq_list
)












if __name__=="__main__":

    mpr_dict = rgbt234.MPR(seqs=seq_list)

    print('')
    for k,v in mpr_dict.items():
        print(k, "MPR", round(v[0]*100, 1))

    print('')
    msr_dict = rgbt234.MSR(seqs=seq_list)

    for k,v in msr_dict.items():
        print(k, "MSR", round(v[0]*100, 1))
