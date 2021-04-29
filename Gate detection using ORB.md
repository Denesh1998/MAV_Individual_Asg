# Gate detection using ORB

## Introduction

We implement an algorithm for detecting corners of gates in images taken by drones in a racing course.This README contains instructions for reproducing  results shown in the report.


## Requirements

Python3
opencv
numpy
sklearn

## Getting Started

Clone this repository by using the following command:

```git clone https://github.com/Denesh1998/MAV_Individual_Asg.git ```

Each image has the path "WashtingtonOBRace/WashtingtonOBRace/img_x.png"

We first illustrate the results of the algorithm on a single image. Run the file single_detection.py. This file is sufficient to illustrate the algorithm working on a single image. Change 'x' to the numbers present in the image file names for viewing the result on different images. The following images would be displayed for the default value of x = 404. 

![](https://)![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_73b3c7420e50cb03f250551101d1c382.png)
![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_61c50982d49b71832b7b976aba995b08.png)
## Considered images

The images that were considered for getting the positive detections plot  have the numbers 
x = {8,47,68,110,124,136,150,172,186,192,198,205,234,239,246,254,256,274,285,291,298,304,315,326,328,373,383,387,397,404,415,422,433}

## Getting results
Run main.py to generate the rest of the plots shown in the report.


