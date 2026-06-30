

import rgbt     
from seqList import *
lasher = rgbt.LasHeR()


result_path= ""
seq_list,length = where_seq_already(result_path, prefix="")
print("seq num: ", length)



lasher(
    tracker_name="tracker_name1",
    result_path=result_path,
    seqs=seq_list
)






if __name__=="__main__":

    mpr_dict = lasher.PR(seqs=seq_list)

    print('')
    for k,v in mpr_dict.items():
        print(k, "PR", round(v[0]*100, 1))


    

    
    
    


    print('')
    msr_dict = lasher.SR(seqs=seq_list)

    for k,v in msr_dict.items():
        print(k, "SR", round(v[0]*100, 1))
