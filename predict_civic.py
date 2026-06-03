from ultralytics import YOLO

model = YOLO('yolov8n.pt')
# สั่งให้โมเดลตรวจจับภาพ Civic.png ในโฟลเดอร์ และเซฟผลกลับมาที่ฝั่ง Windows
# model.predict(source='Civic.png', device=0, save=True, project='.', name='result')

results = model.predict(
    source='civic.jpg', 
    device=0, 
    save=True, 
    project='/usr/src/app',  # บังคับให้เซฟลงจุดที่เชื่อมกับ Windows ตรงๆ
    name='result'
)
