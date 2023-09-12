import os

from matplotlib import pyplot as plt


def plot():
    res7_bpp = [0.1154, 0.1472, 0.1877, 0.3110]
    res7_psnr = [28.8252, 30.7125, 31.5325, 32.6750]
    res7_ms_ssim = [0.9682, 0.9622, 0.9564, 0.9599]

    res6_bpp = [0.1154, 0.1472, 0.1877, 0.0894]
    res6_psnr = [28.8252, 30.7125, 31.5325, 30.6967]
    res6_ms_ssim = [0.9682, 0.9622, 0.9564, 0.9235]

    res5_bpp = [0.1154, 0.1472, 0.1877, 0.03110]
    res5_psnr = [29.0025, 30.7125, 31.5325, 29.0025]
    res5_ms_ssim = [0.9682, 0.9622, 0.9564, 0.8756]

    dvc_bpp = [0.1154, 0.1472, 0.1877, 0.3110]
    dvc_psnr = [28.8252, 30.7125, 31.5325, 33.6102]
    dvc_ms_ssim = [0.9682, 0.9622, 0.9564, 0.9590]

    prefix = 'performance'
    LineWidth = 2
    test, = plt.plot(res7_bpp, res7_psnr, marker='x', color='black', linewidth=LineWidth, label='Proposed')

    # bpp, psnr, msssim = [0.176552, 0.107806, 0.074686, 0.052697], [37.754576, 36.680327, 35.602740, 34.276196], [
    #     0.970477, 0.963935, 0.955738, 0.942226]
    # baseline, = plt.plot(bpp, psnr, "b-*", linewidth=LineWidth, label='baseline')
    #
    # # Ours very fast
    # bpp, psnr, msssim = [0.187701631, 0.122491399, 0.084205003, 0.046558501], [36.52492847, 35.78201761,
    #                                                                            35.05371763, 33.56996097], [
    #     0.968154218, 0.962246563, 0.956369263, 0.942897242]
    # h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')
    #
    # bpp, psnr = [0.165663191, 0.109789007, 0.074090183, 0.039677747], [37.29259129, 36.5842637, 35.88754734,
    #                                                                    34.46536633]
    # h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')

    savepathpsnr = prefix + '/UVG_psnr' + '.png'
    print(prefix)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    # plt.legend(handles=[h264, h265, baseline, test], loc=4)
    plt.legend(handles=[test], loc=4)
    plt.grid()
    plt.xlabel('Bpp')
    plt.ylabel('PSNR')
    plt.title('UVG dataset')
    plt.savefig(savepathpsnr)
    plt.clf()

if __name__ == '__main__':
    plot()