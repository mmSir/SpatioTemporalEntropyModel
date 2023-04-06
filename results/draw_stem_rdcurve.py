import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import imageio
import cv2
import numpy as np
from mplfonts import use_font


font = {'family': 'serif', 'weight': 'normal', 'size': 9}
matplotlib.rc('font', **font)
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题


LineWidth = 1
MarkerSize = 3

prefix = 'rd_results'

def drawUVG():

    # Ours default
    bpp, psnr, msssim = [0.360398059, 0.148765645, 0.076027148, 0.044918456], [38.17135327, 36.64284159, 35.23609537,
                                                                               33.81775541], [0.978673172, 0.967889817,
                                                                                              0.95755343, 0.945528742]
    x264, = plt.plot(bpp, psnr, "--v", color="gray", linewidth=LineWidth, markersize=MarkerSize,
                     label='$\mathrm{x264}$')  # very fast

    bpp, psnr, msssim = [0.28976166, 0.123933107, 0.058072298, 0.032210368], [38.23271652, 36.77568687, 35.3981028,
                                                                              33.96352277], [0.976551152, 0.966845419,
                                                                                             0.956987797, 0.945751638]
    x265, = plt.plot(bpp, psnr, "--v", color="k", linewidth=LineWidth, markersize=MarkerSize, label='$\mathrm{x265}$')  # very fast

    bpp, psnr = [0.185334726, 0.108594607, 0.076628071, 0.06013501], [37.69306179, 36.68785979, 35.52100014, 34.54747739]
    DVC, = plt.plot(bpp, psnr, "y--o", linewidth=LineWidth + 1, markersize=MarkerSize + 1, label='DVC')
    # bpp, psnr, msssim = [0.1580652024, 0.09510494048, 0.06334845238, 0.05001147619], [37.83633036, 36.77129217, 35.7142323, 35.03949596], [0.971793119, 0.9658603452, 0.9588654881, 0.9534102738]
    # DVCp, = plt.plot(bpp, psnr, "y-o", color="peru", linewidth=LineWidth, label='DVC++')
    bpp, psnr = [0.0524, 0.0781, 0.1278, 0.2303], [34.5390, 35.8618, 37.1899, 38.4590]
    RY, = plt.plot(bpp, psnr, "c--o", color="darkred", linewidth=LineWidth, markersize=MarkerSize, label='HLVC')

    bpp = [0.020140105078928754, 0.030122591945569703, 0.047460595461998455, 0.06795096322016461, 0.11418563922646605, 0.22241681262860083, 0.37162872154706794]
    psnr = [31.8495,33.0860,34.4086,35.8495,36.9355,38.3871,39.3656]
    SSF, = plt.plot(bpp, psnr, "c--o", color="peru", linewidth=LineWidth, markersize=MarkerSize, label='SSF')

    bpp, psnr = [0.02709702142196456, 0.04098199165086189, 0.06370908663667266, 0.10385984934969908], [
        34.011881197066536, 35.20922313871838, 36.35551458086286, 37.444169357844764]
    DCVC, = plt.plot(bpp, psnr, "--o", color="hotpink", linewidth=LineWidth, label='DCVC', markersize=MarkerSize)

    bpp, psnr = [0.034, 0.052, 0.080, 0.135], [35.42, 36.48, 37.54, 38.47]
    # dcvc_plus, = plt.plot(bpp, psnr, "-o", color="orange", linewidth=LineWidth, markersize=MarkerSize, label='Temporal Context Mining') # GOP 12

    bpp, psnr = [0.046, 0.062, 0.078, 0.100, 0.120, 0.170], [34.15, 35.21, 36.0, 36.60, 37.124,
                                                             37.90]  # stem baseline from paper
    STEM_paper, = plt.plot(bpp, psnr, '--o', color="orange", linewidth=LineWidth, markersize=MarkerSize, label='stem_paper')

    # stem_roi_baseline
    bpp, psnr = [0.047, 0.083, 0.105, 0.161], [34.384, 36.162, 37.103, 38.314]  # lmbda 0.004 0.010 0.016 0.031
    STEM_ROI_SingleRate_Baseline, = plt.plot(bpp, psnr, '--o', color="red", linewidth=LineWidth + 1,
                                             markersize=MarkerSize + 1, label='stem_baseline')

    psnr = [33.480, 33.943, 34.864, 35.717, 36.338, 36.850, 37.607, 37.986 ,38.488 ]
    bpp = [0.041, 0.046, 0.056, 0.070, 0.082, 0.098, 0.130, 0.160, 0.208]
    STEM_ROI_scGain, = plt.plot(bpp, psnr, '-o', color="b", linewidth=LineWidth + 1, markersize=MarkerSize + 1,
                                label='stem_roi')

    savepathpsnr = prefix + '/UVG_psnr'
    print(prefix)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    # plt.legend(handles=[STEM_ROI_SingleRate_Baseline, STEM_ROI_SingleRate_Baseline_paper, dcvc_plus], bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.legend(handles=[STEM_ROI_scGain, STEM_ROI_SingleRate_Baseline, DVC, RY, DCVC, SSF, STEM_paper, x264, x265 ],
               loc=4, prop={'size': 10.5})
    plt.grid()
    plt.xlabel('Rate(bits per pixel)')
    plt.ylabel('PSNR(dB)')
    plt.title('UVG dataset')
    plt.savefig(savepathpsnr + '.png', dpi=300, bbox_inches='tight')
    plt.clf()

def drawHEVCB():
    bpp, psnr = [0.0688, 0.1093, 0.1883, 0.3536], [31.7031, 33.2380, 34.5965, 35.7882]
    RY, = plt.plot(bpp, psnr, "c--o", color="darkred", linewidth=LineWidth, markersize=MarkerSize, label='HLVC')

    bpp = [0.07658514, 0.11616374, 0.16643274, 0.2939271]
    psnr = [31.82619106, 33.0220837, 34.10751308, 35.10115456]
    DVC, = plt.plot(bpp, psnr, "y--o", linewidth=LineWidth, markersize=MarkerSize, label='DVC')

    bpp, psnr = [0.040289277833824355, 0.061040674080699686, 0.09433236599713564, 0.15265098729928334], [
        31.616849353790283, 32.84782929992676, 33.94570527267456, 34.87294593811035]
    DCVC, = plt.plot(bpp, psnr, "c--o", color="hotpink", linewidth=LineWidth, markersize=MarkerSize, label='DCVC')

    bpp, psnr = [0.038, 0.055, 0.08, 0.115, 0.155, 0.195, 0.245, 0.36, 0.51], \
                [29.5, 30.55, 31.7, 32.75, 33.55, 34.2, 34.75, 35.7, 36.3]  # stem baseline from paper
    STEM_paper, = plt.plot(bpp, psnr, '--o', color="orange", linewidth=LineWidth, markersize=MarkerSize,
                           label='stem_paper')

    bpp = [0.569167464, 0.219282161, 0.110183846, 0.065360897]
    psnr = [35.79301018, 34.03927013, 32.59008599, 31.1074001]
    msssim = [0.977265628, 0.967603596, 0.957326897, 0.94278379]
    x264, = plt.plot(bpp, psnr, "--v", color="gray", linewidth=LineWidth, markersize=MarkerSize, label='$\mathrm{x264}$')  # very fast

    bpp = [0.473727262, 0.185593416, 0.08897321, 0.048943254]
    psnr = [35.78820838, 34.14234791, 32.7354592, 31.24880809]
    msssim = [0.975606114, 0.966591679, 0.956803834, 0.942605916]
    x265, = plt.plot(bpp, psnr, "--v", color="k", linewidth=LineWidth, markersize=MarkerSize, label='$\mathrm{x265}$')  # very fast

    bpp = [0.076, 0.137, 0.181, 0.282] # GOP12
    psnr = [31.993, 33.678, 34.539, 35.636]
    STEM_ROI_SingleRate_Baseline, = plt.plot(bpp, psnr, '--o', color="red", linewidth=LineWidth + 1,
                                             markersize=MarkerSize + 1, label='stem_basline')
    # bpp = [0.076, 0.139, 0.182, 0.284] # GOP10
    # psnr = [31.998, 33.682, 34.541, 35.637]

    bpp = [0.063 , 0.072 , 0.085 , 0.101 , 0.122 , 0.148 , 0.180 , 0.221 , 0.270 , 0.330 , 0.384]
    psnr = [31.135 , 31.669 , 32.217 , 32.745 , 33.258 , 33.744 , 34.244 , 34.692 , 35.077 , 35.412 , 35.664]
    STEM_ROI_scGain, = plt.plot(bpp, psnr, '-o', color="b", linewidth=LineWidth + 1, markersize=MarkerSize + 1,
                                label='stem_roi')



    savepathpsnr = prefix + '/HEVC_B_psnr'
    print(prefix)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    # plt.legend(handles=[STEM_ROI_SingleRate_Baseline, STEM_ROI_SingleRate_Baseline_paper, dcvc_plus], bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.legend(handles=[STEM_ROI_scGain, STEM_ROI_SingleRate_Baseline, DVC, RY, DCVC, STEM_paper, x264, x265],
               loc=4, prop={'size': 10.5})
    plt.grid()
    plt.xlabel('Rate(bits per pixel)')
    plt.ylabel('PSNR(dB)')
    plt.title('HEVC Class B dataset')
    plt.savefig(savepathpsnr + '.png', dpi=300, bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
    plt.figure(figsize=(5.5, 4.5))
    drawUVG()
    drawHEVCB()
