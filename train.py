import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('F:/wzq_deeplearning_project/YOLO11-improve-mask-old/ultralytics/cfg/models/11/yolo11-pose-gfpn_C3k2-mine.yaml')
    # model.load('./runs/train/exp168/weights/best.pt') # loading pretrain weights
    model.train(data='F:/wzq_deeplearning_project/YOLO11-improve-mask-old/ultralytics/cfg/datasets/coco-pose.yaml',
                cache=False,
                imgsz=640,
                epochs=1800,
                batch=16,
                close_mosaic=0,
                workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                device='0',
                optimizer='SGD', # using SGD
                patience=50, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )