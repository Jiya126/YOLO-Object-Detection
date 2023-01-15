import cv2
import numpy as np

cap = cv2.VideoCapture(0)

dim = 320
confThresh = 0.5
nmsThresh = 0.2
classIds = []
confidenceScores = []
boundingBoxes = []


dataClasses = 'coco.names'
dataClassesNames =[]

with open(dataClasses, 'rt') as f:
    dataClassesNames = f.read().rstrip('\n').split('\n')

config = 'yolov3.cfg'
weights = ''

net = cv2.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObject(outputs, img):

    wo, ho, co = img.shape

    for output in outputs:
        for wholeList in output:
            scores = wholeList[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThresh :
                w,h = int(wholeList[2]*wo), int(wholeList[3]*ho)
                x,y = int((wholeList[0]*wo) - w/2), int((wholeList[1]*ho) - h/2)

                boundingBoxes.append([x,y,w,h])
                confidenceScores.append([confidence])
                classIds.append(classId)

    indices = cv2.dnn.NMSBoxes(boundingBoxes, confidenceScores, confThresh, nmsThresh)

    for i in indices:
        i = i[0]
        box = boundingBoxes[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,26,50), 2)
        cv2.putText(img, f'{dataClassesNames[classIds[i]].upper()}  {int(confidenceScores[i]*100)}%', (x+5, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,26,50), 2)




while True:
    suc, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1/255, (dim, dim), [0,0,0], 1)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputLayersNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputLayersNames)

    findObject(outputs, img)

    cv2.imshow('Output' ,img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()