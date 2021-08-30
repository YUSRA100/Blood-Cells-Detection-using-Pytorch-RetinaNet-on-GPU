# pytorch-retinanet

## CSV DataSet Creation


## Training with CSV DataSet

The network is trained using the `train.py` script. For training using a custom dataset, with annotations in CSV format, use

```
python train.py --dataset csv --csv_train <path/to/train_annots.csv>  --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv>
```
Note that the --csv_val argument is optional, in which case no validation will be performed.

## Pre-trained model

A pre-trained model is available at: 
- https://drive.google.com/open?id=1yLmjq3JtXi841yXWBxst0coAgR26MNBS (this is a pytorch state dict)

The state dict model can be loaded using:
```
retinanet = model.resnet50(num_classes=dataset_train.num_classes(),)
retinanet.load_state_dict(torch.load(PATH_TO_WEIGHTS))
```

## Visualization

To visualize the network detection, use `visualize.py`:
```
python visualize.py --dataset coco --coco_path ../coco --model <path/to/model.pt>
```
This will visualize bounding boxes on the validation set. To visualise with a CSV dataset, use:
```
python visualize.py --dataset csv --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv> --model <path/to/model.pt>
```
## Model

The retinanet model uses a resnet backbone. You can set the depth of the resnet model using the --depth argument. Depth must be one of 18, 34, 50, 101 or 152. Note that deeper models are more accurate but are slower and use more memory.


