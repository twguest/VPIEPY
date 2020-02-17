from matplotlib import pyplot as plt
import numpy as np


def plot_obj(confiv, obj):

    fig_amp = plt.figure(figsize=[12, 12])

    plt.set_cmap("bone")

    ax1 = fig_amp.add_subplot(221)
    ax2 = fig_amp.add_subplot(222)
    ax3 = fig_amp.add_subplot(223)
    ax4 = fig_amp.add_subplot(224)

    axes = [ax1, ax2, ax3, ax4]

    ax1.annotate(r"$T_{xx}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w")
    ax2.annotate(r"$T_{yx}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w")
    ax3.annotate(r"$T_{xy}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w")
    ax4.annotate(r"$T_{yy}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w")

    im = ax1.imshow(np.real(obj[0, 0,])**2)
    ax2.imshow(np.real(obj[1, 0,])**2)
    ax3.imshow(np.real(obj[0, 1,])**2)
    ax4.imshow(np.real(obj[1, 1,])**2)

    fig_amp.colorbar(im, ax=axes)
    #######################################################

    fig_phs = plt.figure(figsize=[12, 12])

    plt.set_cmap("bone")

    ax1 = fig_phs.add_subplot(221)
    ax2 = fig_phs.add_subplot(222)
    ax3 = fig_phs.add_subplot(223)
    ax4 = fig_phs.add_subplot(224)

    axes = [ax1, ax2, ax3, ax4]

    ax1.annotate(r"$T_{xx}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w")
    ax2.annotate(r"$T_{yx}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w")
    ax3.annotate(r"$T_{xy}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w")
    ax4.annotate(r"$T_{yy}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w")

    im = ax1.imshow(np.angle(obj[0, 0,]))
    ax2.imshow(np.angle(obj[1, 0,]))
    ax3.imshow(np.angle(obj[0, 1,]))
    ax4.imshow(np.angle(obj[1, 1,]))

    fig_phs.colorbar(im, ax=axes)

    # fig_phs.tight_layout()
    plt.show()


def plot_probe(config, probe):

    fig_amp = plt.figure(figsize=[8, 12])

    plt.set_cmap("bone")

    ax1 = fig_amp.add_subplot(321)
    ax2 = fig_amp.add_subplot(322)
    ax3 = fig_amp.add_subplot(323)
    ax4 = fig_amp.add_subplot(324)
    ax5 = fig_amp.add_subplot(325)
    ax6 = fig_amp.add_subplot(326)

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    ax1.annotate(
        r"$\psi_{1,x}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w"
    )
    ax2.annotate(
        r"$\psi_{1,y}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w"
    )
    ax3.annotate(
        r"$\psi_{2,x}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w"
    )
    ax4.annotate(
        r"$\psi_{2,y}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w"
    )
    ax5.annotate(
        r"$\psi_{3,x}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w"
    )
    ax6.annotate(
        r"$\psi_{3,y}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w"
    )

    im = ax1.imshow(np.real(probe[0, 0,])**2)
    ax2.imshow(np.real(probe[0, 1,]**2))
    ax3.imshow(np.real(probe[1, 0,]**2))
    ax4.imshow(np.real(probe[1, 1,]**2))
    ax5.imshow(np.real(probe[2, 0,]**2))
    ax6.imshow(np.real(probe[2, 1,]**2))

    fig_amp.colorbar(im, ax=axes)
    #######################################################

    fig_phs = plt.figure(figsize=[8, 12])

    plt.set_cmap("twilight_shifted_r")

    ax1 = fig_phs.add_subplot(321)
    ax2 = fig_phs.add_subplot(322)
    ax3 = fig_phs.add_subplot(323)
    ax4 = fig_phs.add_subplot(324)
    ax5 = fig_phs.add_subplot(325)
    ax6 = fig_phs.add_subplot(326)

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    ax1.annotate(
        r"$\psi_{1,x}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w"
    )
    ax2.annotate(
        r"$\psi_{1,y}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w"
    )
    ax3.annotate(
        r"$\psi_{2,x}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w"
    )
    ax4.annotate(
        r"$\psi_{2,y}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w"
    )
    ax5.annotate(
        r"$\psi_{3,x}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w"
    )
    ax6.annotate(
        r"$\psi_{3,y}$", (0.75, 0.10), textcoords="axes fraction", size=25, c="w"
    )

    im = ax1.imshow(np.angle(probe[0, 0,]))
    ax2.imshow(np.angle(probe[0, 1,]))
    ax3.imshow(np.angle(probe[1, 0,]))
    ax4.imshow(np.angle(probe[1, 1,]))
    ax5.imshow(np.angle(probe[2, 0,]))
    ax6.imshow(np.angle(probe[2, 1,]))

    fig_phs.colorbar(im, ax=axes)

    # fig_phs.tight_layout()
    plt.show()
