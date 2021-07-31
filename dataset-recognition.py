import tensorflow as tf
import glob
import cv2
import re
import os
def cropFrontView(imgs,labels,idx):
  #read the label.txt file 
  print("")
  xMin= min(labels[0][0],labels[1][0]) 
  xMax= max(labels[2][0],labels[3][0])
  yMin= min(labels[0][1],labels[3][1])
  yMax= max(labels[1][1],labels[2][1])
  h = int(yMax) - int(yMin)
  w = int(xMax) - int(xMin)
  im_cropped = imgs[max((int(yMin) - 2*int(w)),0) : min((int(yMax) + int(w)),imgs.shape[0]), max((int(xMin) - 2*int(w),0)) : min((int(xMax) + 2*int(w)),imgs.shape[1])]
  cv2.imwrite("carDataSabok/Img"+str(idx)+".jpg", im_cropped)

def extractLabel(labels):
  point_str = labels.readline()
  x = re.split("\[", point_str)
  y = re.split("{", x[1])
  x1 = re.split("-",y[1])[0]
  y1 = re.split("-",y[1][:-1])[1]
  print(x1,y1)
  x2 = re.split("-",y[2])[0]
  y2 = re.split("-",y[2][:-1])[1]
  print(x2,y2)
  x3 = re.split("-",y[3])[0]
  y3 = re.split("-",y[3][:-1])[1]
  print(x3,y3)
  x4 = re.split("-",y[4])[0]
  y4 = re.split("-",y[4][:-2])[1]
  print(x4,y4)
  points = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
  return points
def makeDatasetFromPaltes(im_list,label_list):
  print("loading the images")
  for idx in range(len(im_list)):
    sizeXY = im_list[idx].shape
    cropFrontView(im_list[idx],label_list[idx],idx)

def main():
  print("start making dataset by croping a big margin from located plates.")
  im_list = []
  label_list = []
  for img in glob.glob("D:/LPR_Projects/vehicle-detection-Recognition/data/*.jpg"):
    if(os.path.exists(img.replace('jpg','txt'))):
      im_list.append(cv2.imread(img))
      label_list.append(extractLabel(open(img.replace('jpg','txt'), "r")))
  makeDatasetFromPaltes(im_list,label_list)

if __name__ == "__main__":
    main()