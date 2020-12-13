'''
 Author: Mark Landergan 2020 
 Author: Austin Scott 2020
'''
import argparse, csv
import cv2, imutils
import datetime, time, threading, random
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import os
 
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--wait-time", type=int, default=15, help="minimum time between detections")
ap.add_argument("-b", "--base-update-time", type=int, default=60, help="time between updating base frame")
ap.add_argument("-c", "--min-area", type=int, default=500, help="minimum area size")
ap.add_argument("-d", "--search-width", type=int, default=200, help="width of downscaled search image")

args = vars(ap.parse_args())

class MotionDetection:
	
	def __init__(self, display_detections = True):
		self._vs = WebcamVideoStream(src=0).start()
		self._fps = FPS().start()
		self._camera_error = 0
		self._base_frame = None
		self._current_frame = None
		self._search_frame = None
		self._search_gray_frame = None
		self._display_detections = display_detections
		self._default_img = cv2.imread("lib/default_img.png")

		try:
			if self._vs.frame == None:
				self._camera_error = 1
		except:
			pass

	# call read function on WebcamVideoStream object - if no stream is available, read default image
	def grab(self):
		if self._camera_error == 1:
			self._current_frame = self._default_img
		else:
			self._current_frame = self._vs.read()
		return self._current_frame

	# resize the current frame 
	def resize(self):
		self._search_frame = imutils.resize(self._current_frame, width = args["search_width"]) # use downscaled _search_frame to do detection - but save full resolution images from _current_frame
		# no need to crop frame
		# height, width, channels = self._search_frame.shape 
		# self._search_frame = self._search_frame[0:height, int(width/3):width]

	# convert the current frame to grayscale and blur it
	def gray_blur(self):
		self._search_gray_frame = cv2.cvtColor(self._search_frame, cv2.COLOR_BGR2GRAY)
		self._search_gray_frame = cv2.GaussianBlur(self._search_gray_frame, (21,21), 0)

	# Update base frame every 5 minutes
	def update_base_frame(self): 
		if(self._search_gray_frame is None):
			print("[INFO] updateBaseFrame called when gray frame wasn't set")
			self._current_frame = self.grab()
			self.resize()
			self.gray_blur()
		self._base_frame = self._search_gray_frame
		print(time.ctime())
		threading.Timer(args["base_update_time"], self.update_base_frame).start()

	# calculate contours between base frame and current frame
	def calculate_area_diff(self):
		# compute the absolute difference between the current frame 
		# and base frame
		frameDelta = cv2.absdiff(self._base_frame, self._search_gray_frame)
		thresh = cv2.threshold(frameDelta, 40, 255, cv2.THRESH_BINARY)[1]
		
		# dilate the thresholded image to fill in the holes, then find 
		# contours on thresholded image
		thresh = cv2.dilate(thresh, None, iterations=2)
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		
		return [frameDelta, thresh, cnts]
	
	def display_images(self, frameDelta):
		cv2.imshow("Current frame", md._current_frame)
		cv2.imshow("Gray search frame", md._search_gray_frame)
		cv2.imshow("Gray base frame", md._base_frame)
		cv2.imshow("Area difference", frameDelta)

		key = cv2.waitKey(1) & 0xFF
		return key

	def bird_or_nah(self):
		# return yah for now - this is where we can use our neural network to classify the image
		return True

	# get a random name from database
	def name_bird(self):
		index = int(random.random()*2000) # there is 2000 random names in bird_names.csv
		with open('lib/bird_names.csv', newline='') as f:
			reader = csv.reader(f)
			count = 0
			for row in reader:
				if (count == index):
					return row[0]
				else:
					count = count+1
	
	def addAnnotations(self):
		text = "Occupied"
		
		# draw the text and timestamp on the frame
		cv2.putText(md._current_frame, "Bird Detected: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.putText(md._current_frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, md._current_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)	

	# rescale and pad roi for _current_frame based on detected roi from _search_image
	# this reduces search time, but maximizes image quality
	def create_upscaled_roi(self, x, y, w, h):
		height, width, channels = self._current_frame.shape 
		scale = width/args["search_width"]
		x = x * scale - width*0.05
		x = int(max(0, x))
		y = y * scale - height*0.05
		y = int(max(0, y))
		w = w * scale + width*0.1
		w = int(min(width - x, w))
		h = h * scale + height*0.1
		h = int(min(height - y, h))
		return (x, y, w, h)

if __name__=="__main__":
	md = MotionDetection()
	md.update_base_frame()
	
	random.seed()

	last_time = time.time()
	count = 0

	while(True):
		md.grab()
		md.resize()				
		md.gray_blur()
		frameDelta, thresh, cnts = md.calculate_area_diff()
	
		# loop over the contours
		for c in cnts:
			# if the contour is too small, ignore it
			if cv2.contourArea(c) < args["min_area"]:
				continue
			
			# compute the bounding box for the contour
			(x, y, w, h) = cv2.boundingRect(c)
			(x, y, w, h) = md.create_upscaled_roi(x, y, w, h) # create new bounding rect that includes padding around bounding box and is upscaled for full res image
			cv2.rectangle(md._current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			roi = md._current_frame[y:y+h,x:x+w]
			
			#check to see if the passed image is in fact a bird using neural network
			if(md.bird_or_nah): # TODO - currently just a shell function

				# limit amount of saved photos
				if((time.time() - last_time) >= args["wait_time"]):
					if not os.path.exists('photos'):
						os.makedirs('photos')
					
					bird_name = md.name_bird()
					cv2.imwrite("photos/%s_%s_%d.jpg" % (bird_name, datetime.datetime.now().strftime("%d-%m-%Y"), count), roi)
					count +=1
					last_time = time.time()
			
		# show the frame and record if the user presses a key
		key = md.display_images(frameDelta)			
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


			
	
