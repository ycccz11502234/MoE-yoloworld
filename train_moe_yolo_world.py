from ultralytics import YOLOWorld
from ultralytics import YOLOWorld_MoE
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
from ultralytics.models.yolo.world_moe.train_world import WorldTrainerFromScratchMoE

data = dict(
    train=dict(
        yolo_data=["Objects365.yaml"],
        grounding_data=[
            dict(
                img_path="datasets/flickr30k/images",
                json_file="datasets/flickr30k/final_flickr_separateGT_train.json",
            ),
            dict(
                img_path="datasets/GQA/images",
                json_file="datasets/GQA/final_mixed_train_no_coco.json",
            ),
        ],
    ),
    val=dict(yolo_data=["lvis.yaml"]),
)
model = YOLOWorld("yolov8s-worldv2.yaml")
model.train(data=data, epochs=100, device=[-1], project = "runs/yoloworld_officialresult", name = "origin", batch=64, trainer=WorldTrainerFromScratch)


model = YOLOWorld("yolov8-worldv2-moe-final.yaml")
model.train(data=data, epochs=100, device=[-1], project = "runs/moe_yoloworld", name = "moe_yoloworld", batch=64, trainer=WorldTrainerFromScratch)
