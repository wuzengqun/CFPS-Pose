import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('runs/train/exp92/weights/best.pt') # 选择训练好的权重路径
    model = YOLO('model/faster+GSFFPN+LDEDH.pt') # 选择训练好的权重路径
    model.val(data='F:/wzq_deeplearning_project/YOLO11-improve-mask-old/ultralytics/cfg/datasets/coco-pose.yaml',
              split='test', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=32,
              # iou=0.7,
              # rect=False,
              device=0,
              save_json=True, # if you need to cal coco metrice
              project='runs/test',
              name='exp',
              )