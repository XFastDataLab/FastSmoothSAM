from ultralytics import YOLO

model = YOLO(model="yolov8s-seg.yaml", \
             )
model.train(data="ultralytics/datasets/sa.yaml", \
            epochs=100, \
            batch=32, \
            imgsz=640, \
            save=True, \
            save_period=10, \
            device='0,1,2,3,4,5,6,7', \
            project='sa_test', \
            # val=False, \
            # resume=True,\
            )
