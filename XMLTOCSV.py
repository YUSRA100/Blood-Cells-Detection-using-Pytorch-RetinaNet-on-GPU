

import os
import xml.etree.ElementTree as ET
import random
import math
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', type=str)
    parser.add_argument('-p', '--percent', type=float, default=0.2)
    parser.add_argument('-t', '--train', type=str, default='train.csv')
    parser.add_argument('-v', '--val', type=str, default='test.csv')
    parser.add_argument('-c', '--classes', type=str, default='class.csv')
    args = parser.parse_args()
    return args

#Get a list of files with a specific suffix
def get_file_index(indir, postfix):
    file_list = []
    for root, dirs, files in os.walk(indir):
        for name in files:
            if postfix in name:
                file_list.append(os.path.join(root, name))
    return file_list

#Write label information
def convert_annotation(csv, address_list):
    cls_list = []
    with open(csv, 'w') as f:
        for i, address in enumerate(address_list):
            in_file = open(address, encoding='utf8')
            strXml =in_file.read()
            in_file.close()
            root=ET.XML(strXml)
            for obj in root.iter('object'):
                cls = obj.find('name').text
                cls_list.append(cls)
                xmlbox = obj.find('bndbox')
                b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                     int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
                f.write(file_dict[address_list[i]])
                f.write( "," + ",".join([str(a) for a in b]) + ',' + cls)
                f.write('\n')
    return cls_list


# if __name__ == "__main__":
args = parse_args()
file_address = args.indir
test_percent = args.percent
train_csv = args.train
test_csv = args.val
class_csv = args.classes
#path_i = 'C:\\Users\\ASDF\\Downloads\\OPPD-master-DATA-images_full-PAPRH\\Images'#'C:\\Users\\ASDF\\Downloads\\LS-SSDD-v1.0-OPEN\\Annotations_sub'
#path_a = 'C:\\Users\\ASDF\\Downloads\\OPPD-master-DATA-images_full-PAPRH\\XML'#C:\\Users\\ASDF\\Downloads\\LS-SSDD-v1.0-OPEN\\JPEGImages_sub'
path = 'D:\\Github\\Pytorch-retinanet\\BCCD'#D:\\Videos\\FLIR\\Self-crafted\\Person'#'C:\\Users\\ASDF\\Downloads\\ocr\\FINAL_DATASET\\AUG_IMG'
Annotations = get_file_index(path, '.xml')
    #'C:\\Users\\ASDF\\Downloads\\ocr\\FINAL_DATASET', '.xml')
    # 'D:\\Videos\\Clip14-AnnotatedData', '.xml')  
Annotations.sort()
# ext = ['.png','.jpg']
# JPEGfiles = get_file_index('C:\\Users\\ASDF\\Downloads\\ocr\\FINAL_DATASET','.png' ) #Can be modified according to the image suffix of your own dataset. This is the path of the JPEGImage of your VOC dataset. The path of each person is different. I usually use the absolute path
JPEGfiles = get_file_index(path,'.jpg' )
# JPEGfiles.extend(JPEGfiles2)
JPEGfiles.sort()
assert len(Annotations) == len(JPEGfiles) #If the XML file and the image file name cannot be one-to-one correspondence, an error will be reported
file_dict = dict(zip(Annotations, JPEGfiles))
num = len(Annotations)
test = random.sample(k=math.ceil(num*test_percent), population=Annotations)
train = list(set(Annotations) - set(test))

cls_list1 = convert_annotation(train_csv, train)
cls_list2 = convert_annotation(test_csv, test)
cls_unique = list(set(cls_list1+cls_list2))

with open(class_csv, 'w') as f:
    for i, cls in enumerate(cls_unique):
        f.write(cls + ',' + str(i) + '\n')
