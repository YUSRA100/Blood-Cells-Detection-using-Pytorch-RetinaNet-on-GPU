# Implementation-pytorch-retinanet-BCCD-Dataset

## CSV DataSet Creation

BCCD dataset has been used and uploaded with XML annotations. This implementation takes CSV files so 'XMLtoCSV.py' file takes folder of dataset having Images and XML's and creates train.csv, test.csv and class.csv. The python script can also be customized with percentage of training and testing data. The created files with be then provided to the model for training and testing.

## Training with CSV DataSet

The network is trained using the `train.py` script. For training using a custom dataset, with annotations in CSV format, use
```
python train.py --dataset csv --csv_train <path/to/train_annots.csv>  --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv>
```
The --csv_val argument is optional and can be skipped. 
```
python train.py --dataset csv --csv_train <path/to/train_annots.csv>  --csv_classes <path/to/train/class_list.csv>
```
## Testing
For testing, I have utilized three methods: 
- To test/visulaize the network detection using test.csv file, use `visualize.py`
```
python visualize.py --dataset csv --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv> --model <path/to/model.pt>
```

- To test on sample images from local directory rather than from random test csv samples, use visulaize_single_image.py
``` 
python visualize_single_image.py --image_dir "dir path" --model_path "model.pt path" --class_list "class.csv path" 
```

- To visualize results on a video sample, use 'visualize_video_run.py'. This will also save an output video with bounding boxes results. 

## Pre-trained model
A pre-trained model is available at: 
- https://drive.google.com/open?id=1yLmjq3JtXi841yXWBxst0coAgR26MNBS (this is a pytorch state dict)
The state dict model can be loaded using:
```
retinanet = model.resnet50(num_classes=dataset_train.num_classes(),)
retinanet.load_state_dict(torch.load(PATH_TO_WEIGHTS))
```
The retinanet model uses a resnet backbone. You can set the depth of the resnet model using the --depth argument. Depth must be one of 18, 34, 50, 101 or 152. Note that deeper models are more accurate but are slower and use more memory.
