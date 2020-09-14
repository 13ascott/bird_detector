'''
 Author: Mark Landergan 2020 
'''
import argparse
import cv2, imutils
import datetime, time, threading
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import os

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--wait-time", type=int, default=5, help="minimum time between detections")
ap.add_argument("-b", "--base-update-time", type=int, default=5, help="time between updating base frame")
ap.add_argument("-c", "--min-area", type=int, default=500, help="minimum area size")

args = vars(ap.parse_args())

class MotionDetection:
	
	def __init__(self, display_detections = True):
		self._count = 0
		self._vs = WebcamVideoStream(src=0).start()
		self._fps = FPS().start()
		self._base_frame = None
		self._current_frame = None
		self._current_gray_frame = None
		self._display_detections = display_detections

	# resize the current frame 
	def resize(self):
		self._current_frame = imutils.resize(self._current_frame, width=500)
		height, width, channels = self._current_frame.shape
		self._current_frame = self._current_frame[0:height, int(width/3):width]

	# convert the current frame to grayscale and blur it
	def grayBlur(self):
		self._current_gray_frame = cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2GRAY)
		self._current_gray_frame = cv2.GaussianBlur(self._current_gray_frame, (21,21), 0)

	# Update base frame every 5 minutes
	def updateBaseFrame(self):
		if(self._current_gray_frame is None):
			print("[INFO] updateBaseFrame called when gray frame wasn't set")
			self._current_frame = md._vs.read()
			self.resize()
			self.grayBlur()
		self._base_frame = self._current_gray_frame
		print(time.ctime())
		threading.Timer(args["base_update_time"], self.updateBaseFrame).start()

	# calculate contours between base frame and current frame
	def calculateAreaDiff(self):
		# compute the absolute difference between the current frame 
		# and base frame
		frameDelta = cv2.absdiff(self._base_frame, self._current_gray_frame)
		thresh = cv2.threshold(frameDelta, 80, 255, cv2.THRESH_BINARY)[1]
		
		# dilate the thresholded image to fill in the holes, then find 
		# contours on thresholded image
		thresh = cv2.dilate(thresh, None, iterations=2)
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		
		return [frameDelta, thresh, cnts]
	
	def displayImages(self, frameDelta):
		cv2.imshow("Current frame", md._current_frame)
		cv2.imshow("Gray base frame", md._base_frame)
		cv2.imshow("Area difference", frameDelta)

		key = cv2.waitKey(1) & 0xFF
		return key
	
	def addAnnotations(self):
		text = "Occupied"
		
		# draw the text and timestamp on the frame
		cv2.putText(md._current_frame, "Bird Detected: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.putText(md._current_frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, md._current_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)	

if __name__=="__main__":
	md = MotionDetection()
	
	md.updateBaseFrame()
	
	last_time = time.time()
	count = 0

	while(True):
		md._current_frame = md._vs.read()
		md.resize()				
		md.grayBlur()
		frameDelta, thresh, cnts = md.calculateAreaDiff()
	
		# loop over the contours
		for c in cnts:
			# if the contour is too small, ignore it
			if cv2.contourArea(c) < args["min_area"]:
				continue
			
			# compute the bounding box for the contour
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(md._current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			roi = md._current_frame[y:y+h,x:x+w]
		     
		    # limit amount of saved photos
			if((time.time() - last_time) >= args["wait_time"]):
				if not os.path.exists('photos'):
					os.makedirs('photos')
	
				cv2.imwrite("photos/frame%d.jpg" % count, roi)
				count +=1
				last_time = time.time()
			
			# show the frame and record if the user presses a key
			key = md.displayImages(frameDelta)			
			# if the `q` key is pressed, break from the lop
			if key == ord("q"):
				break
			
			md._fps.update()
			
	# stop the timer and dispaly FPS info
	md._fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(md._fps.elapsed()))
	print("[INFO] approx, FPS: {:.2f}".format(md._fps.fps()))

	cv2.destroyAllWindows()
	md._vs.stop()	


			
	
