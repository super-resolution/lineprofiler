from controllers import processing_microtubule, processing_SNC, processing_one_channel


def process_factory(mode, microscope_image, config, fit_func, tear_down, progress_bar, ID):
    if mode == "Microtubule":
        process = processing_microtubule.QProcessThread()
    elif mode == "SNC":
        process =  processing_SNC.QProcessThread()
    elif mode == "SNC one channel":
        process = processing_one_channel.QProcessThread()
    else:
        raise ValueError("Unknown mode")
    process.set_data(ID, microscope_image.data, microscope_image.file_path)
    if microscope_image.data_z is not None:
        process.data_z = microscope_image.data_z
    process.sig.connect(progress_bar.setValue)
    for k in config.keys():
        setattr(process, k, config[k])
    process.sig_plot_data.connect(fit_func)
    process.done.connect(tear_down)
    return process