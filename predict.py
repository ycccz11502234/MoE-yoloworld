from ultralytics import YOLOWorld_MoE
from ultralytics import YOLOWorld


# Initialize a YOLO-World model
model = YOLOWorld_MoE("weights/yolov8x-worldv2.pt")  # or choose yolov8m/l-world.pt

model.set_classes(["bus", "person"])


results = model.predict(
    "test_image/000000002006.jpg",
    save=True,            # save visualized result image
    save_txt=False,       # set True to also dump YOLO-format txt labels
    save_conf=False,      # include confidence in saved txt (only with save_txt=True)
    conf=0.25,            # confidence threshold
    iou=0.7,              # NMS IoU threshold
    project="runs/predict",
    name="zebra",
    exist_ok=True,        # overwrite the same folder instead of creating zebra2/zebra3...
)

for r in results:
    print(r.boxes)
    print("saved to:", r.save_dir)
