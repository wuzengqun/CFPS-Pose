import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    print("🔥 val.py started")
    model = YOLO('model/fasternet+GSFFPN.pt')
    print("✅ YOLO model loaded")
    model.val(   
        data='F:/wzq_deeplearning_project/YOLO11-improve-mask-old/ultralytics/cfg/datasets/coco-pose.yaml',
        split='val',
        imgsz=640,
        batch=32,
        device=0,
        save_json=True,
        project='runs/val',
        name='exp'
    )
