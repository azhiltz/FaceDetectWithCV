#!/usr/bin/python
# -*- coding: UTF-8 -*-

import cv2 
import argparse
import sys
import numpy as np

from FaceDetector_cv import CVFaceDetector
import unittest


def drawPred( frame, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        label = '%.2f' % conf

        # Print a label of class.
        label = 'Face'

        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

class TestFaceDetector( unittest.TestCase ):
	def setUp( self ):
		self.prototxt = 'model/deploy.prototxt'
		self.weights = 'model/model_weights.caffemodel'
		self.detector = CVFaceDetector( proto_path=self.prototxt, weights_path=self.weights, det_conf_thresh=0.6 )

	def tearDown(self):
		pass

	def test_detection( self ):
		img = cv2.imread('data/test.jpg' )
		self.assertTrue( img is not None )
		result = self.detector.detect_face( img )
		self.assertTrue( len(result) > 0 )
		print(result)

		for i in range(len(result)):
			r = result[i]
			drawPred( img, r[0], r[1], r[2], r[1]+r[3], r[2]+r[4] )
		cv2.imshow("result", img )
		cv2.imwrite("data/result.jpg", img )
		cv2.waitKey(10)

if __name__ == '__main__':
	unittest.main()