from ultralytics import YOLO

def main():
    # 1. โหลดโมเดลตั้งต้น (แนะนำ yolov8n.pt ขนาดเล็กและเทรนไว หรือ yolov8s.pt ถ้าต้องการความแม่นยำเพิ่มขึ้น)
    model = YOLO('yolov8n.pt')

    # 2. สั่งเทรนโมเดล
    results = model.train(
        data='/usr/src/app/data.yaml',   
        epochs=100,                      
        imgsz=640,                       
        device=0,                        
        project='/usr/src/app',          
        name='car_model_result',
        exist_ok=True          
    )

if __name__ == '__main__':
    main()