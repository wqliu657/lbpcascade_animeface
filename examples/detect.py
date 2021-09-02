import cv2
import sys
import os.path

filepath = "/home/wqliu/Workplace/lbpcascade_animeface/examples/inputimage"
fileoutpath = "/home/wqliu/Workplace/lbpcascade_animeface/examples/outimage"
imagepath = os.listdir(filepath)

def detect(image1,filename, cascade_file = "../lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)
    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 3,
                                     minSize = (24, 24))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #cv2.imshow("AnimeFaceDetect", image)
    #cv2.waitKey(0)
    if(len(faces) == 0):
        a=1
    else:
        out = os.path.join(fileoutpath,image1)
        cv2.imwrite(out, image)


for images in imagepath:
    image = os.path.join(filepath,images)
    detect(images,image)
#if len(sys.argv) != 2:
    #sys.stderr.write("usage: detect.py <filename>\n")
    #sys.exit(-1)
    
#detect(sys.argv[1])
