# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

#====USER CONSTANTS=====
A = 34.87
Distance = 0
#Shloks's constant:
# m=1 for y=1
# m=1.13 for y>1
#=======================


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_grid(img, grid_shape, color=(0, 255, 0), thickness=1):
    h, w, _ = img.shape
    rows, cols = (15,15)
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
		
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
#===Second Screen
vs2 = VideoStream(src=1).start()
#================

time.sleep(2.0)
fps = FPS().start()
#===Second Screen
fps2 = FPS().start()
#================

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=500, height=300)
	#===Second Screen
	frame2 = vs2.read()
	frame2 = imutils.resize(frame2, width=500, height=300)
	#================

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
	#===Second Screen
	(h2, w2) = frame2.shape[:2]
	blob2 = cv2.dnn.blobFromImage(cv2.resize(frame2, (300, 300)),
		0.007843, (300, 300), 127.5)
	#================

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	#===Second Screen
	net.setInput(blob2)
	detections2 = net.forward()
	#================

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]
		confidence2 = detections2[0, 0, i, 2]
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"] or confidence2 > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			#===Second Screen
			idx_2 = int(detections2[0, 0, i, 1])
			box_2 = detections2[0, 0, i, 3:7] * np.array([w2, h2, w2, h2])
			(startX_2, startY_2, endX_2, endY_2) = box_2.astype("int")
			#================

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			#===Second Screen
			label2 = "{}: {:.2f}%".format(CLASSES[idx_2],
				confidence2 * 100)
			#================

			No_Of_Boxes_Covered = (abs(startY-endY)/20)+1
			No_Of_Boxes_Covered_2 = (abs(startY_2-endY_2)/20)+1
			if(CLASSES[idx] == "person"):
				cv2.putText(frame, 'Human Presence Detected', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
				if No_Of_Boxes_Covered==1:
					ShloksConstant = 1
				else:
					ShloksConstant = 1.13
				Distance = (A*ShloksConstant)/No_Of_Boxes_Covered
				Distance = "%.2f"%Distance
				cv2.putText(frame, "Distance = " + str(Distance)+ " meters", (40,110), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
			#===Second Screen
			if(CLASSES[idx_2] == "person"):
				cv2.putText(frame2, 'Human Presence Detected', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
				if No_Of_Boxes_Covered_2==1:
					ShloksConstant = 1
				else:
					ShloksConstant = 1.13
				Distance_2 = (A*ShloksConstant)/No_Of_Boxes_Covered_2
				Distance_2 = "%.2f"%Distance_2
				cv2.putText(frame2, "Distance = " + str(Distance_2)+ " meters", (40,110), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
			#================

			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			#===Second Screen	
			cv2.rectangle(frame2, (startX_2, startY_2), (endX_2, endY_2),
				COLORS[idx_2], 2)
			#================
			
			#===Second Screen
			y = startY - 15 if startY - 15 > 15 else startY + 15
			y2 = startY_2 - 15 if startY_2 - 15 > 15 else startY_2 + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			cv2.putText(frame2, label2, (startX_2, y2),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx_2], 2)
			#================
			
			
	# show the output frame
	
	
	frame = draw_grid(frame, (300,300))
	#===Second Screen
	frame2 = draw_grid(frame2, (300,300))
	#================

	cv2.imshow("Feed 1", frame)
	#===Second Screen
	cv2.imshow("Feed 2", frame2)
	#================


	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()
	#===Second Screen
	fps2.update()
	#================

# stop the timer and display FPS information
fps.stop()
print("[INFO] FEED_1 elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] FEED_1 approx. FPS: {:.2f}".format(fps.fps()))
#===Second Screen
fps2.stop()
print("[INFO] FEED_2 elapsed time: {:.2f}".format(fps2.elapsed()))
print("[INFO] FEED_2 approx. FPS: {:.2f}".format(fps2.fps()))
#================

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
