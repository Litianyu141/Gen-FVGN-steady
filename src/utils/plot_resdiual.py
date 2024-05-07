import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")
import numpy as np
from matplotlib.ticker import AutoMinorLocator

# 加载CSV文件
CFD_file_path = '/mnt/c/Users/DOOMDUKE2-lab/Desktop/GPE-FVGN-local/results_compare/cavity_flow/Re=400-Cavity-starccm-resdiual.csv'  # 请将此路径替换为您的CSV文件路径
FVGN_file_path = '/mnt/c/Users/DOOMDUKE2-lab/Desktop/GPE-FVGN-local/results_compare/cavity_flow/cavity_Re=400mean_u4_mu0.01_500steps_finetune(False)_15/NO.15_mean_u4_rho_1_mu_0.01_source0_dt_1_aoa_0_500steps_finetune(False)_residuals.csv'  # 请将此路径替换为您的CSV文件路径

CFD_res = pd.read_csv(CFD_file_path)
FVGN_res = pd.read_csv(FVGN_file_path)

# 定义一个函数来绘制两个残差曲线图
def plot_residuals(CFD_res, FVGN_res, max_iteration=None):
    fig, axs = plt.subplots(1, 1, figsize=(7, 3.5))

    # 计算共同的y轴界限
    if max_iteration is not None:
        CFD_res = CFD_res[CFD_res['迭代'] <= max_iteration]
        FVGN_res = FVGN_res[CFD_res['迭代'] <= max_iteration]

    min_y = min(CFD_res.iloc[:, 1:].min().min(), FVGN_res.min().min())
    max_y = max(CFD_res.iloc[:, 1:].max().max(), FVGN_res.max().max())
    
    # 设置CFD残差曲线（点划线）
    for column in CFD_res.columns[1:]:
        axs.plot(CFD_res['迭代'], CFD_res[column], linestyle='-', label='CFD: ' + column)

    # 设置FVGN残差曲线（实线）
    for column in FVGN_res.columns[0:]:
        axs.plot(CFD_res['迭代'], FVGN_res[column], label='Gen-FVGN: ' + column)

    # axs.set_ylim([min_y, max_y])
    axs.set_xlim([0, max(CFD_res['迭代'].max(), CFD_res['迭代'].max())])  # 设置x轴界限
    axs.set_xlabel('Iteration Steps',fontsize=10)
    axs.set_ylabel('Residual Value (Log Scale)',fontsize=10)
    axs.set_yscale('log')

    # 设置对数刻度的y轴的次要刻度
    log_loc = matplotlib.ticker.LogLocator(subs=np.arange(2, 10))
    axs.yaxis.set_minor_locator(log_loc)

    # 增大x轴和y轴刻度标签的字体大小，并调整刻度长度
    axs.tick_params(axis='x', labelsize=8, length=6)  # Major ticks for x-axis
    axs.tick_params(axis='y', which='major', labelsize=8, length=6)  # Major ticks for y-axis
    axs.tick_params(axis='y', which='minor', length=4)  # Minor ticks for y-axis


    axs.legend(fontsize=8)
    plt.subplots_adjust(bottom=0.2)  # 调整子图布局以向上移动整个图表
    # 显示图表
    plt.show()
    plt.savefig("/home/doomduke2/GEP-FVGN/GEP-FVGN/repos-py/FVM/my_FVNN/utils/Gen-FVGN-results/cavity_re=400_residuals.png",dpi=500)
    plt.close()
    
# 调用函数示例（例如仅绘制前20步的数据）
plot_residuals(CFD_res,FVGN_res, max_iteration=500)

