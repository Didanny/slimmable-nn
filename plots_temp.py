import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import scienceplots as sp
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

column_width = 3.5

with plt.style.context(['science']):
    sci_cycle = plt.rcParams['axes.prop_cycle']
    
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
# SHOW = False
SHOW = True

df_cifar = pd.read_csv('./results/us_results_cifar.csv')
df_cifar = df_cifar.melt(id_vars=['Model Name'], var_name='sparsity', value_name='value')
df_cifar['sparsity'] = df_cifar['sparsity'].astype('float64') * 100
df_cifar['value'] = df_cifar['value'] * 100
df_cifar['Model Name'] = df_cifar['Model Name'].astype('string')
df_cifar = df_cifar.sort_values(by=['Model Name', 'sparsity'])
pattern_vgg = r'vgg'
pattern_resnet = r'resnet'

# Filter rows where "Model Name" matches the regex pattern
df_cifar_vgg = df_cifar[df_cifar['Model Name'].str.contains(pattern_vgg, regex=True, case=False)]
df_cifar_resnet = df_cifar[df_cifar['Model Name'].str.contains(pattern_resnet, regex=True, case=False)]
df_cifar_resnet = df_cifar_resnet.drop_duplicates(subset=['value'], keep='last')

df_cifar_latency = pd.read_csv('./results/us_latency_results_cifar_b1_cpu.csv')
df_cifar_latency = df_cifar_latency.melt(id_vars=['Model Name'], var_name='sparsity', value_name='latency')
df_cifar_latency['sparsity'] = df_cifar_latency['sparsity'].astype('float64') * 100
df_cifar_latency['Model Name'] = df_cifar_latency['Model Name'].astype('string')
df_cifar_latency = df_cifar_latency.sort_values(by=['Model Name', 'sparsity'])

df_accuracy_resnet56 = df_cifar_resnet[df_cifar_resnet['Model Name'] == 'usresnet56']
df_accuracy_vgg16 = df_cifar_vgg[df_cifar_vgg['Model Name'] == 'usvgg16_bn']

df_latency_resnet56 = df_cifar_latency[df_cifar_latency['Model Name'] == 'usresnet56']
df_latency_vgg16 = df_cifar_latency[df_cifar_latency['Model Name'] == 'usvgg16_bn']

resnet56_sparsities = df_accuracy_resnet56['sparsity'].tolist()

df_latency_resnet56 = df_latency_resnet56[df_latency_resnet56['sparsity'].isin(resnet56_sparsities)]

df_resnet56 = pd.merge(df_accuracy_resnet56, df_latency_resnet56, on='sparsity')
df_vgg16 = pd.merge(df_accuracy_vgg16, df_latency_vgg16, on='sparsity')

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Plotting
with plt.style.context(['science', 'ieee']):
    plt.rcParams['axes.prop_cycle'] = sci_cycle
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(figsize=(column_width * 1.87, column_width/1.5), ncols=2)
    
    # ResNet plot
    sns.lineplot(data=df_cifar_resnet, x='sparsity', y='value', hue='Model Name', marker='o', markersize=3, ax=ax1)
    ax1.set_xlabel('Width (\%)')
    ax1.set_ylabel('Accuracy (\%)')
    plt.sca(ax1)
    plt.xticks([25, 50, 75, 100])
    
    # VGG plot
    sns.lineplot(data=df_cifar_vgg, x='sparsity', y='value', hue='Model Name', marker='o', markersize=3, ax=ax2)
    ax2.set_xlabel('Width (\%)')
    ax2.set_ylabel('')
    plt.sca(ax2)
    plt.xticks([25, 50, 75, 100])  
    
    # ResNet legend
    model_names = ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56']
    legend_handles = [Line2D([0], [0], color=sci_cycle.by_key()['color'][i], marker='o', markersize=3, mec='#ffffff', mew=0.8, label=model_names[i]) for i in range(4)]
    ax1.legend(handles=legend_handles, loc='lower right')
    
    # VGG legend
    model_names = ['VGG11', 'VGG13', 'VGG16', 'VGG19']
    legend_handles = [Line2D([0], [0], color=sci_cycle.by_key()['color'][i], marker='o', markersize=3, mec='#ffffff', mew=0.8, label=model_names[i]) for i in range(4)]
    ax2.legend(handles=legend_handles, loc='lower right')
    
    if SHOW:
        plt.show()
    else:
        plt.savefig('./results/plots/uscifar_accuracy.pgf', bbox_inches='tight')
        plt.close()
    
    
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

with plt.style.context(['science', 'ieee']):
    plt.rcParams['axes.prop_cycle'] = sci_cycle
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(figsize=(column_width * 1.87, column_width/1.5), ncols=2)
    
    # ResNet56
    inference_time_cpu_resnet56 = sns.lineplot(data=df_resnet56, x='sparsity', y='latency', marker='o', c=sci_cycle.by_key()['color'][6], ax=ax1, linewidth=1, markersize=3)
    # ax1.set_ylim(50, 300)
    
    ax1.set_ylabel('CPU Inference Time (ms)')
    ax1.set_xlabel('Width (\%)')
    
    ax1_flip = ax1.twinx()
    
    accuracy_resnet56 = sns.lineplot(data=df_resnet56, x='sparsity', y='value', marker='o', c=sci_cycle.by_key()['color'][1], ax=ax1_flip, linewidth=1, markersize=3)
    # ax1_flip.set_ylim(0, 40)
    
    # ax1_flip.set_ylabel('Accuracy (\%)')
    
    # VGG16
    inference_time_cpu_vgg16 = sns.lineplot(data=df_vgg16, x='sparsity', y='latency', marker='o', c=sci_cycle.by_key()['color'][6], ax=ax2, linewidth=1, markersize=3)
    # ax2.set_ylim(50, 300)
    
    ax2.set_ylabel('')
    ax2.set_xlabel('Width (\%)')
    
    ax2_flip = ax2.twinx()
    
    accuracy_vgg16 = sns.lineplot(data=df_vgg16, x='sparsity', y='value', marker='o', c=sci_cycle.by_key()['color'][1], ax=ax2_flip, linewidth=1, markersize=3)
    # ax2_flip.set_ylim(0, 40)
    
    ax2_flip.set_ylabel('Accuracy (\%)')
    
    # legend_handles = [Line2D([0], [0], color=sci_cycle.by_key()['color'][6], marker='o', markersize=3, mec='#ffffff', mew=0.8, label='Inference Time'),
    #                   Line2D([0], [0], color=sci_cycle.by_key()['color'][1], marker='o', markersize=3, mec='#ffffff', mew=0.8, label='mAP@50'),
    # ]
    
    # fig.legend(handles=legend_handles, loc='lower right', bbox_to_anchor=(0.9, 0.1))
    
    if SHOW:
        plt.show()
    else:
        plt.savefig('./results/plots/usyolov5n.pgf', bbox_inches='tight')
        plt.close()

pass