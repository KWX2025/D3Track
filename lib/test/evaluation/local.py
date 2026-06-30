from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.gtot_path = '/data1/andong.lu/data/RGBT_DATA/GTOT/'
    settings.lasher_path = '/media/ha/2T/datasets/lasher/testingset/'
    settings.lashertestingset_path = '/media/ha/2T/datasets/lasher/testingset/'
    settings.network_path = '/media/ha/2T/Hanv/code/4090/CKD-master/lib/test/networks/'    # Where tracking networks are stored.
    settings.prj_dir = './'
    settings.result_plot_path = '/media/ha/2T/Hanv/code/4090/CKD-master/lib/test/result_plots/'
    settings.results_path = '/media/ha/2T/kwx/CKD(IGF+四帧模板更新+交叉蒸馏bast+优化门控）最好）/output/train_dq_27_4kabast/'    # Where to store tracking results
    settings.rgbt210_path = '/data1/andong.lu/data/RGBT_DATA/RGBT210/'
    settings.rgbt234_path = '/media/ha/2T/datasets/RGB_T234/'
    settings.vtuav_path = '/home/ha/download/dataset'
    settings.save_dir = './'
    settings.segmentation_path = '/media/ha/2T/Hanv/code/4090/CKD-master/lib/test/segmentation_results/'

    return settings

