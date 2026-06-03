import torch
from ultralytics import YOLO

# 1. ตรวจสอบสถานะการเชื่อมต่อ GPU ของ PyTorch ข้างใน Docker
print("=" * 50)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print("=" * 50)

# 2. เรียกใช้โมเดล YOLOv8 รุ่น Nano (ดาวน์โหลดออโต้)
model = YOLO('yolov8n.pt') 

# 3. สั่งประมวลผลรูปภาพตัวอย่าง โดยบังคับให้วิ่งบน GPU (device=0) และเซฟผลลัพธ์
print("กำลังประมวลผลภาพบน GPU...")
results = model.predict(source='https://ultralytics.com/images/bus.jpg', device=0, save=True, project='.')

print("\n--- สำเร็จ! ประมวลผลเสร็จสิ้นบน GPU เรียบร้อยแล้ว ---")