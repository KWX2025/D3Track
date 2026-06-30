"""
TODO: 填入结果文件的路径信息，评估跟踪器
"""

import rgbt     # pip install rgbt==1.0.1   github: https://github.com/opacity-black/RGBT_toolkit
from seqList import *
lasher = rgbt.LasHeR()


result_path= "/media/ha/2T/kwx/CKD(IGF+四帧模板更新+交叉蒸馏bast+优化门控）最好）/output/train_dq_27_4kabast/dftrack/dftrack_4b_dropmae_dftrack_tf_cc_mask.25_ep030/lasher"  #
seq_list,length = where_seq_already(result_path, prefix="")
print("seq num: ", length)



lasher(
    tracker_name="tracker_name1",
    result_path=result_path,
    seqs=seq_list
)


# 画图
# lasher.draw_attributeRadar(metric_fun=lasher.MPR, filename="eval_tracker/CAiATrack_RGBT234_MPR.png")
# lasher.draw_attributeRadar(metric_fun=lasher.MSR, filename="eval_tracker/CAiATrack_RGBT234_MSR.png")

if __name__=="__main__":

    mpr_dict = lasher.PR(seqs=seq_list)

    print('')
    for k,v in mpr_dict.items():
        print(k, "PR", round(v[0]*100, 1))


    # mpr_dict = lasher.NPR(seqs=seq_list)

    # print('')
    # for k,v in mpr_dict.items():
    #     print(k, "NPR", round(v[0]*100, 1))


    print('')
    msr_dict = lasher.SR(seqs=seq_list)

    for k,v in msr_dict.items():
        print(k, "SR", round(v[0]*100, 1))
