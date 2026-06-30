from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    settings.gtot_path = ''
    settings.lasher_path = ''
    settings.lashertestingset_path = ''
    settings.network_path = ''
    settings.prj_dir = './'
    settings.result_plot_path = ''
    settings.results_path = './output'
    settings.rgbt210_path = ''
    settings.rgbt234_path = ''
    settings.vtuav_path = ''
    settings.save_dir = './'
    settings.segmentation_path = ''

    return settings
