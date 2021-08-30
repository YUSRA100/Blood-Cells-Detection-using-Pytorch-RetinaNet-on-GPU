import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
from collections import Counter

def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image(image_path, model_path, class_list):

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    model = torch.load(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()
    labellist = []
    # for img_name in os.listdir(image_path):
        # print("Image nam e is" ,img_name)
    # image = cv2.imread(image_path)
    image = image_path
    image_orig = image.copy()
    rows, cols, cns = image.shape
    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
    rows, cols, cns = image.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = new_image.astype(np.float32)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))

    with torch.no_grad():

        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()

        st = time.time()
        print(image.shape, image_orig.shape, scale)
        scores, classification, transformed_anchors = model(image.cuda().float())
        print('Elapsed time: {}'.format(time.time() - st))
        idxs = np.where(scores.cpu() > 0.5)

        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]

            x1 = int(bbox[0] / scale)
            y1 = int(bbox[1] / scale)
            x2 = int(bbox[2] / scale)
            y2 = int(bbox[3] / scale)
            label_name = labels[int(classification[idxs[0][j]])]
            print(bbox, classification.shape)
            score = scores[j]
            caption = '{} {:.3f}'.format(label_name, score)
            # draw_caption(img, (x1, y1, x2, y2), label_name)
            draw_caption(image_orig, (x1, y1, x2, y2), caption)
            cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            print(label_name)
            counter = None
            labellist.append(label_name)
        counter = Counter(labellist)                
        counter = dict(counter)
        color = (0,255,0) #green
        if len(counter)==0:
            cv2.putText(image_orig, "NO OBJECTS DETTECTED", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:            
            if "WBC" in counter:    
                    cv2.putText(image_orig, "WBC : "+ str(counter["WBC"]),  (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.putText(image_orig, "WBC : "+ str(0),  (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if "RBC" in counter:    
                    cv2.putText(image_orig, "RBC : "+ str(counter["RBC"]),  (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.putText(image_orig, "RBC : "+ str(0),  (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) 
            if "Platelets" in counter:    
                    cv2.putText(image_orig, "Platelets : "+ str(counter["Platelets"]),  (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.putText(image_orig, "Platelets : "+ str(0),  (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)                                                
        labellist = []
        return image_orig
        
if __name__ == '__main__':
    totalFrames = 0
    vid = "video.mp4" #Path to directory containing video
    model_path =  "model_final.pt" #'Path to model
    class_list = "class.csv"#Path to CSV file listing class names (see README)
    start_time = time.time()
    cap = cv2.VideoCapture(vid)
    while (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')
        break
    # get the frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # define codec and create VideoWriter object 
    out = cv2.VideoWriter("output_clip.mp4", 
                          cv2.VideoWriter_fourcc(*'mp4v'), 20, 
                          (frame_width, frame_height))
    frame_count = 0 # to count total frames
    total_fps = 0 # to get the final frames per second
    
    while(cap.isOpened()== True):
    # capture each frame of the video
        ret, frame = cap.read()
        if ret == False:
            print("breaking bad")
            break
        
        if totalFrames % 3 ==0:            
            print('---------------------video reading begins---------------------------')    
            image_orig = detect_image(frame, model_path, class_list)
            print('---------------------Sending output ---------------------------')
            # get the end time
        frame_count += 1
        totalFrames += 1        
        # press `q` to exit
        cv2.imshow('image', image_orig)
        if out is not None:
            out.write(image_orig)
                    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break        
    if out is not None:
        out.release()
    end_time = time.time()
    # get the fps
    fps = 1 / (end_time - start_time)
    # add fps to total fps
    total_fps += fps
    # increment frame count
    print("....The fps are : ", fps)
    print("....The total frames detected are : ",frame_count)
    cv2.destroyAllWindows()
    cap.release()
        
        