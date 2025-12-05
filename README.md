The user guide will be released after acceptance. Please stay tuned and be patient.

# 行为识别系统（CFPS-Pose+Bytetrack+2s-agcn）  
本仓库包含了CFPS-Pose源码

演示视频-Jetson Xavier NX
---
<video src="https://github.com/user-attachments/assets/20126a7a-dd91-4e99-a359-0c832dcf45d0" controls width="400">
  Your browser does not support the video tag.
</video>  

演示视频-4090
---
<video src="https://github.com/user-attachments/assets/cf4af6c3-c0d9-42ff-bf1c-56a2e8e824f7" controls width="400">
  Your browser does not support the video tag.
</video>  


数据集准备  
---
一、COCO2017数据集  
├─coco-pose  
│  ├─annotations  
│  ├─images  
│  │  ├─train2017  
│  │  └─val2017  
│  └─labels  
│      ├─train2017  
│      └─val2017  

二、CrowdPose  
首先到官网下载数据集及Annotations：https://github.com/jeffffffli/CrowdPose?tab=readme-ov-file  
随后解压，并使用spilt_pic.py将数据集划分为训练集、验证集与测试集。最后通过to_yolo.py将数据集转为yolo格式，供本文模型训练。  
└─crowdpose  
    ├─images  
    │  ├─test  
    │  ├─train  
    │  └─val   
    └─labels  
    │  ├─test  
    │  ├─train  
    │  └─val  
    ├─spilt_pic.py  
    ├─to_yolo.py  


  训练   
  ---
以COCO数据集为例：


