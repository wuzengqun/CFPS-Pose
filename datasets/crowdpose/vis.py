import cv2
import numpy as np

def visualize_annotation(image_path, annotation_line, img_size=640):
    # 解析标注
    parts = list(map(float, annotation_line.split()))
    class_id = int(parts[0])
    bbox = parts[1:5]  # x_center, y_center, w, h
    keypoints = np.array(parts[5:]).reshape(-1, 3)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
        
    h, w = image.shape[:2]
    
    # 绘制边界框
    x_center, y_center, bw, bh = bbox
    x1 = int((x_center - bw/2) * w)
    y1 = int((y_center - bh/2) * h)
    x2 = int((x_center + bw/2) * w)
    y2 = int((y_center + bh/2) * h)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 绘制关键点
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 0),
              (0, 128, 0), (0, 0, 128), (128, 128, 0),
              (128, 0, 128), (0, 128, 128), (128, 128, 128),
              (255, 128, 0)]
    
    for i, (x, y, v) in enumerate(keypoints):
        if v > 0:  # 只绘制可见的关键点
            px = int(x * w)
            py = int(y * h)
            color = colors[i % len(colors)]
            cv2.circle(image, (px, py), 4, color, -1)
            cv2.putText(image, str(i), (px+5, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # 显示图像
    cv2.imshow('Annotation Check', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用示例
image_path = "C:/pycharm_project/YOLO11-improve-mask/datasets/crowdpose/images/val/100018.jpg"  # 替换为对应的图像路径
annotation_line = "0 0.344703 0.806077 0.189406 0.387845 0.362500 0.749415 2 0.287500 0.711944 1 0.381250 0.800937 2 0.304688 0.714286 1 0.404687 0.829040 2 0.331250 0.744731 2 0.370312 0.843091 1 0.295312 0.864169 2 0.420312 0.885246 2 0.315625 0.932084 2 0.415625 0.967213 2 0.000000 0.000000 0 0.334375 0.632319 1 0.320312 0.737705 1"
visualize_annotation(image_path, annotation_line)