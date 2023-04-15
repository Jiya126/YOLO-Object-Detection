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

# print(dataClassesNames)

config = 'yolov3-tiny.cfg'
weights = 'yolov3-tiny.weights'

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
                confidenceScores.append(float(confidence))
                classIds.append(classId)
            # print(len(boundingBoxes))
    # print(output)
    # print(len(output))

    indices = cv2.dnn.NMSBoxes(boundingBoxes, confidenceScores, confThresh, nmsThresh)

    # print(indices)

    for i in indices:
        # print(i)
        # i = i[0]
        box = boundingBoxes[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,26,50), 2)
        cv2.putText(img, f'{dataClassesNames[classIds[i]].upper()}  {int(confidenceScores[i]*100)}%', (x+5, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,26,50), 2)




while True:
    suc, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1/255, (dim, dim), [0,0,0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputLayersNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputLayersNames)
    # print(outputLayersNames)
    # print(outputs[0].shape)

    findObject(outputs, img)

    cv2.imshow('Output' ,img)
    if cv2.waitKey(1)==ord('q'):
        break
    # cv2.waitKey(1)
    
cap.release()
cv2.destroyAllWindows()