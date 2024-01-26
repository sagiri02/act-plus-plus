# 复现mobile-aloha工作

本文基于[mobile-aloha的开源代码](https://github.com/MarkFzp/act-plus-plus)复现部分工作，按照从浅到深，包含三大部分的内容：
- Step1：单纯的跑通代码
- Step2：ACT算法和实现细节讲解，并且最大化的复现论文中的实验结果
- Step3：关于该论文的一些思考

**注意：本文仅仅包含mobile-aloha软件方面的结果复现，也就说基于仿真数据完成策略的学习，不涉及任何硬件和实际平台的内容。**

# 跑通代码
首先需要下载仓库源代码，链接：https://github.com/MarkFzp/act-plus-plus。
由于源代码有一些小错误，或者交代不清，因此我对它做了一些修改。参考代码仓库：https://github.com/huiwenzhang/act-plus-plus。嫌麻烦的朋友，也可以直接pull这个仓库代码。相比于源仓库主要的改动有：
- 增加了`requirements.txt`编译依赖
- 修复了部分运行错误
- 增加了部分shell脚本，方便测试，位于`scripts`文件下
- 如果需要可视化或者测试实际的数据，还需要用到[aloha-hardware](https://github.com/MarkFzp/mobile-aloha) 中的部分代码，主要是`aloha_scripts`文件下的一些脚本，因此本仓库直接将该文件夹复制到了根目录下。

**下面的复现过程都是基于改动后的仓库，请

## 依赖安装
仓库中部分依赖阐述不是很清楚，作者后续做了补充。我新建了一个requirements.txt文件，内容如下：

```
torchvision
torch
pyquaternion
pyyaml
rospkg
pexpect
mujoco==2.3.7
dm_control==1.0.14
opencv-python
matplotlib
einops
packaging
h5py
ipython
wandb
robomimic
diffusers
```

使用者可以通过`pip install -r requirements.txt`安装相关依赖。

其他部分代码中有些错误，大家可参照`https://github.com/MarkFzp/act-plus-plus/issues`的内容查阅。本代码已经做了修改，所以也可以直接pull本仓库的代码。

## 数据集准备
数据集分为两部分，分别是实际采集的数据和仿真数据。作者提供下载地址：
- 实际数据：https://drive.google.com/drive/folders/1FP5eakcxQrsHyiWBRDsMRvUfSxeykiDc
- 仿真数据：https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O

注意
- 仿真数据也可以自己生成，具体生成下面会介绍
- 对于实际数据，其实下载下来也没有什么用，因为RL或者模仿学习的训练都是交互的。看了下如果采用实际数据，会创建一个real_env，这个环境是需要和实际物体平台交互的。所以没有平台，代码跑不起来【可能理解有误】

涉及的代码部分如下：
```python
# get task parameters
is_sim = task_name[:4] == 'sim_'
# print teask name and config
print('task_name: ', task_name)
if is_sim or task_name == 'all':
    from constants import SIM_TASK_CONFIGS
    task_config = SIM_TASK_CONFIGS[task_name]
else:
    from aloha_scripts.constants import TASK_CONFIGS
    task_config = TASK_CONFIGS[task_name]
```
可见训练中是通过名字来区分是实际训练还是仿真数据。如果是实际数据，训练中使用的是`real_env.py`，仿真数据和环境交互的代码位于`sim_env.py`。
### 数据集下载

### 可视化
为了可视化数据集，可以使用`scripts/visulize_eps.sh`脚本。用法：
```
# 可视化实际采集数据
./scripts/visualize_eps.sh [task_name] [episode_idx] real
# 可视化仿真数据
./scripts/visualize_eps.sh [task_name] [episode_idx]
```
- task_name：表示任务的名字，比如`sim_transfer_cube`
- episode_idx：表示示教demo的序号


注意对于仿真数据集和实际采集的数据集，要使用不同的脚本来可视化。该仓库中提供的可视化python脚本只适用于仿真数据，也就是`sim_insertation_xxx`和`sim_transfer_cube_xxx`任务。对于下载的实际示教的数据，需要使用`visualize_episodes_real.py`脚本。具体原因参考：https://github.com/MarkFzp/act-plus-plus/issues/16。
实际上对比两个脚本的代码可以发现，对于实际采集的视频数据，经过了压缩处理，因此在加载数据时需要进行解压缩。然后再保存成视频。如果不经过压缩，会发现加载的视频图像的维度是错误，因此在执行concanate操作时，会出错。



### 生成仿真数据
除了利用遥操作来获得操作任务的数据外，还可以利用一些手工编写的策略来产生数据。作者在项目中提供了两个任务，分别是`sim_inertion_scripted`和`sim_transfer_cube_scripted`。生成的数据会保存成hdf5的格式。

- sim_inertion_scripted
![](docs/insert.png)

- sim_transfer_cube_scripted
![](docs/transfer.png)

## 算法训练

## 算法评估


# 算法实现细节解析

变分自编码器参考：https://kexue.fm/archives/5253/comment-page-2
生成模型三大类：
- GAN
- VAE
- diffusion model

## 评估action chunking对于结果的影响


## 评估temporal ensembling的影响


# 一些思考