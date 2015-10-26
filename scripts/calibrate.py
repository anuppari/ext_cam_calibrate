#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 20:53:19 2015

@author: Anup
"""

import rospy
import tf
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import collections

# Quaternion helper function handles
qMult = tf.transformations.quaternion_multiply # quaternion multiplication function handle
qConj = tf.transformations.quaternion_conjugate # quaternion conjugate function handle
q2m = tf.transformations.quaternion_matrix # quaternion to 4x4 transformation matrix

def rotateVec(p,q):
    return qMult(q,qMult(np.append(p,0),qConj(q)))[0:3]


def rvec2quat(rvec):
    theta = np.linalg.norm(rvec)
    u = rvec/theta
    return np.hstack((u*np.sin(theta/2),np.cos(theta/2)))


# Main node
def calibration():
    global bridge, tfl, tfbr, localObjPts
    global K
    global cornerBuff, data, solutionBuff, state
    global squareSize, numHorizCorners, numVertCorners, sampleBuffSize, stationaryThreshold, cameraTF
    
    # Init node
    rospy.init_node('calibrate')
    
    # Get node parameters
    squareSize = rospy.get_param("~squareSize",0.06) # [m]
    zOffset = rospy.get_param("~zOffset",-0.007) # [m]
    numHorizCorners = rospy.get_param("~numHorizCorners",8)
    numVertCorners = rospy.get_param("~numVertCorners",6)
    sampleBuffSize = rospy.get_param("~sampleBuffSize",10)
    stationaryThreshold = rospy.get_param("~stationaryThreshold",1) # [pixels]
    numMeasurements = rospy.get_param("~numMeasurements",5)
    cameraName = rospy.get_param("~cameraName","camera")
    cameraTF = rospy.get_param("~cameraTF","camera")
    
    # Object for converting ROS images to OpenCV images
    bridge = CvBridge()
    
    # Circular buffers for storing 3D and 2D data
    state = 'waiting'
    cornerBuff = collections.deque(maxlen=sampleBuffSize)
    data = {'tIm2Board':collections.deque(maxlen=sampleBuffSize),
            'qIm2Board':collections.deque(maxlen=sampleBuffSize),
            'tCam':collections.deque(maxlen=sampleBuffSize),
            'qCam':collections.deque(maxlen=sampleBuffSize),
            'tBoard':collections.deque(maxlen=sampleBuffSize),
            'qBoard':collections.deque(maxlen=sampleBuffSize)}
    solutionBuff = {'tIm2Cam':[],
                    'qIm2Cam':[]}
    
    # tf handles
    tfl = tf.TransformListener()
    tfbr = tf.TransformBroadcaster()
    
    # Get camera intrinsic parameters and distortion coefficients
    K = None
    camInfoSub = rospy.Subscriber(cameraName+"/camera_info",CameraInfo,camInfoCB)
    rospy.loginfo("Waiting to get camera intrinsic parameters...")
    while (K is None) and (not rospy.is_shutdown()): # Wait until recieved camera info
        rospy.sleep(0.5)
        pass
    camInfoSub.unregister()
    rospy.loginfo("Got camera intrinsic parameters!")
    rospy.loginfo("K: "+str(K))
    rospy.loginfo("D: "+str(D))
    
    # Create array of local object points
    y,x = np.meshgrid(range(numHorizCorners),range(numVertCorners))
    localObjPts = squareSize*np.vstack((x.flatten()+1,y.flatten()+1,np.zeros(numHorizCorners*numVertCorners)+zOffset/squareSize)).T
    
    # Subscribe to camera and mocap topics
    image_sub = rospy.Subscriber(cameraName+"/image_raw",Image,imageCB)
    
    # Wait for shutdown, publish solution
    while not rospy.is_shutdown():
        if (len(solutionBuff['tIm2Cam']) > 0) and (len(solutionBuff['qIm2Cam']) > 0):
            # Publish average solution
            tIm2Cam = np.mean(solutionBuff['tIm2Cam'],axis=0)
            qIm2Cam = np.mean(solutionBuff['qIm2Cam'],axis=0)
            tfbr.sendTransform(tIm2Cam,qIm2Cam,rospy.Time.now(),"image",cameraTF)
            
            # Finish if enough measurements
            if (len(solutionBuff['tIm2Cam']) >= numMeasurements) and (len(solutionBuff['qIm2Cam']) >= numMeasurements):
                rospy.signal_shutdown("Done taking measurements")
    rospy.sleep(0.1)
    
    # Print solution
    np.set_printoptions(precision=15)
    print "tIm2Cam: "+str(solutionBuff['tIm2Cam'])
    print "qIm2Cam: "+str(solutionBuff['qIm2Cam'])
    rospy.loginfo("translation: "+str(np.mean(solutionBuff['tIm2Cam'],axis=0)))
    rospy.loginfo("quaternion: "+str(np.mean(solutionBuff['qIm2Cam'],axis=0)))
    rospy.loginfo("translation variance: "+str(np.var(solutionBuff['tIm2Cam'],axis=0)))
    rospy.loginfo("quaternion variance: "+str(np.var(solutionBuff['qIm2Cam'],axis=0)))
    
    # Close windows
    cv2.destroyAllWindows()


def imageCB(imageMsg):
    global bridge, tfl, tfbr, localObjPts
    global K, D # camera intrinsics
    global state, data, cornerBuff
    
    # image capture time stamp
    imageTimeStamp = imageMsg.header.stamp
    
    # Convert ros image message to opencv image
    try:
        cv_image = bridge.imgmsg_to_cv2(imageMsg,"bgr8")
    except CvBridgeError, e:
        pass
    
    # Check if chessboard is visible, and find corners if visible
    patternSize = (numHorizCorners,numVertCorners)
    (cornersFound,corners) = cv2.findChessboardCorners(cv_image,patternSize,flags=cv2.CALIB_CB_FAST_CHECK)
    corners = np.squeeze(corners)
    
    if cornersFound:
        # Get better corner accuracy
        cv_image_gray = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
        corners = refineCorners(cv_image_gray,corners)
        
        # Show corners on image
        cv2.drawChessboardCorners(cv_image,patternSize,corners,cornersFound)
        cv2.putText(cv_image,state,(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,255)
        cv2.imshow("camera image",cv_image)
        cv2.waitKey(3)
        
        # Get calibration board pose relative to image sensor, expressed in image frame. p_image = rvec*p_board + tvec
        (poseFound,rvec,tvec) = cv2.solvePnP(localObjPts,corners,K,D)
        rvec = np.squeeze(rvec)
        tvec = np.squeeze(tvec)
        q = rvec2quat(rvec)
        
        # Calculate image pose relative to calibration board, expressed in board frame. p_board = qIm2Board*p_image + tIm2Board
        qIm2Board = qConj(q)
        tIm2Board = -1*rotateVec(tvec,qIm2Board)
        tfbr.sendTransform(tIm2Board,qIm2Board,imageTimeStamp,"image2","board")
        
        if state=='waiting':                                        # waiting for camera to stabilize
            if len(cornerBuff) == sampleBuffSize:                   # buffer full, check if camera is stable
                maxPixelMotion = pixelMotion(cornerBuff)            # check pixel motion to see if camera is stationary
                if maxPixelMotion < stationaryThreshold:            # if stationary, clear buffer and start sampling
                    cornerBuff.clear()
                    rospy.sleep(0.5)                                # sleep to overcome mocap delay
                    state = 'sampling'
                else:                                               # else, just add newest data to buffer
                    cornerBuff.append(corners)
            else:                                                   # else, just add newest data to buffer
                cornerBuff.append(corners)
        
        elif state=='sampling':
            if len(cornerBuff) == sampleBuffSize:                   # buffer full, check if camera is stable
                # Average values
                tIm2Board = np.mean(data['tIm2Board'],axis=0)
                qIm2Board = np.mean(data['qIm2Board'],axis=0)
                tBoard = np.mean(data['tBoard'],axis=0)
                qBoard = np.mean(data['qBoard'],axis=0)
                tCam = np.mean(data['tCam'],axis=0)
                qCam = np.mean(data['qCam'],axis=0)
                
                # Calculate image pose w.r.t. world, expressed in world frame. p_world = qIm2World*p_image + tIm2World
                tIm2World = tBoard + rotateVec(tIm2Board,qBoard)
                qIm2World = qMult(qBoard,qIm2Board)
                tfbr.sendTransform(tIm2World,qIm2World,imageTimeStamp,"image","world")
                
                # Calculate image pose w.r.t. camera, expressed in camera frame. p_cam = qIm2Cam*p_image + tIm2Cam
                tIm2Cam = rotateVec(tIm2World-tCam,qConj(qCam))
                qIm2Cam = qMult(qConj(qCam),qIm2World)
                
                # Deal with equivalent representations
                if qIm2Cam[-1] < 0:
                    qIm2Cam = -1*qIm2Cam
                
                # Add to solution buffer
                solutionBuff['tIm2Cam'].append(tIm2Cam)
                solutionBuff['qIm2Cam'].append(qIm2Cam)
                
                # Clear data buffers
                data['tIm2Board'].clear()
                data['qIm2Board'].clear()
                data['tBoard'].clear()
                data['qBoard'].clear()
                data['tCam'].clear()
                data['qCam'].clear()
                state = 'moving'
                
                # Finish if good covariance
                
                
            else:                                                   # else, just add newest data to buffer
                cornerBuff.append(corners)
                
                # Get calibration board pose relative to image sensor, expressed in image frame. p_image = rvec*p_board + tvec
                (poseFound,rvec,tvec) = cv2.solvePnP(localObjPts,corners,K,D)
                rvec = np.squeeze(rvec)
                tvec = np.squeeze(tvec)
                q = rvec2quat(rvec)
                
                # Calculate image pose relative to calibration board, expressed in board frame. p_board = qIm2Board*p_image + tIm2Board
                qIm2Board = qConj(q)
                tIm2Board = -1*rotateVec(tvec,qIm2Board)
                data['qIm2Board'].append(qIm2Board)
                data['tIm2Board'].append(tIm2Board)
                
                # Get calibration board pose w.r.t. world, expressed in world frame. p_world = qBoard*p_board + tBoard
                tfl.waitForTransform("/world","/board",imageTimeStamp,rospy.Duration(0.5))
                (tBoard,qBoard) = tfl.lookupTransform("/world","/board",imageTimeStamp)
                data['tBoard'].append(tBoard)
                data['qBoard'].append(qBoard)
                
                # Get camera pose w.r.t. world, expressed in world frame. p_world = qCam*p_cam + tCam
                tfl.waitForTransform("/world",cameraTF,imageTimeStamp,rospy.Duration(0.5))
                (tCam,qCam) = tfl.lookupTransform("/world",cameraTF,imageTimeStamp)
                data['tCam'].append(tCam)
                data['qCam'].append(qCam)
        
        elif state=='moving':
            cornerBuff.append(corners)                              # add corners to buffer
            if len(cornerBuff) > 1:
                maxPixelMotion = pixelMotion(cornerBuff)            # check pixel motion to see if camera is stationary
                if maxPixelMotion > 10*stationaryThreshold:
                    state = 'waiting'
    else:
        # Show image without corners
        cv2.putText(cv_image,state,(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,255)
        cv2.imshow("camera image",cv_image)
        cv2.waitKey(3)


def refineCorners(monoImage,corners):
    # Based on code in
    # Use a radius of half the minimum distance between corners. This should be large enough to snap to the
    # correct corner, but not so large as to include a wrong corner in the search window.
    
    # Distance between horizontal neighbors
    min_distance = float("inf")
    for row in range(numVertCorners):
        for col in range(numHorizCorners - 1):
            index = row*numVertCorners + col
            min_distance = min(min_distance, np.linalg.norm(corners[index, 0]-corners[index + 1, 0]))
    
    # Distance between vertical neighbors
    for row in range(numVertCorners - 1):
        for col in range(numHorizCorners):
            index = row*numVertCorners + col
            min_distance = min(min_distance, np.linalg.norm(corners[index, 0]-corners[index + numHorizCorners, 0]))
    
    # Minimum overall
    radius = int(np.ceil(min_distance * 0.5))
    cv2.cornerSubPix(monoImage, corners, (radius,radius), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.1))
    
    return corners


def pixelMotion(cornerBuff):
    #numCorners = len(corners)                           # number of checkerboard corners. =numHorizCorners*numVertCorners
    centroids = np.mean(cornerBuff,axis=0) # [np.sum([cornerSet[row] for cornerSet in data['corners']],axis=0)/sampleBuffSize for row in range(numCorners)] # for each checkerboard corner, centroid across buffered images
    errors = cornerBuff - centroids # [[cornerSet[row]-centroids[row] for cornerSet in data['corners']] for row in range(numCorners)] # for each checkerboard corner, error from centroid across buffered images
    avgError = np.mean(np.power(errors,2),axis=0)       # for each checkerboard corner, mean squared error across buffered images
    maxPixelMotion = np.max(avgError)                   # Max across all checkerboard corners and both x and y direction
    
    return maxPixelMotion


def camInfoCB(camInfo):
    global K, D
    D = camInfo.D
    K = np.reshape(np.array(camInfo.K),(3,3))


def stuff():
    # Get calibration board pose relative to image sensor, expressed in image frame. p_image = rvec*p_board + tvec
    (poseFound,rvec,tvec) = cv2.solvePnP(localObjPts,corners,K,D)
    rvec = np.squeeze(rvec)
    tvec = np.squeeze(tvec)
    q = rvec2quat(rvec)
    
    # Calculate image pose relative to calibration board, expressed in board frame. p_board = qIm2Board*p_image + tIm2Board
    qIm2Board = qConj(q)
    tIm2Board = -1*rotateVec(tvec,qIm2Board)
    
    # Get calibration board pose w.r.t. world, expressed in world frame. p_world = qBoard*p_board + tBoard
    tfl.waitForTransform("/world","/board",imageTimeStamp,rospy.Duration(0.5))
    (tBoard,qBoard) = tfl.lookupTransform("/world","/board",imageTimeStamp)
    
    # Calculate image pose w.r.t. world, expressed in world frame. p_world = qIm2World*p_image + tIm2World
    tIm2World = tBoard + rotateVec(tIm2Board,qBoard)
    qIm2World = qMult(qBoard,qIm2Board)
    tfbr.sendTransform(tIm2World,qIm2World,imageTimeStamp,"image","world")
    
    # Get camera pose w.r.t. world, expressed in world frame. p_world = qCam*p_cam + tCam
    tfl.waitForTransform("/world",cameraTF,imageTimeStamp,rospy.Duration(0.5))
    (tCam,qCam) = tfl.lookupTransform("/world",cameraTF,imageTimeStamp)
    
    # Calculate image pose w.r.t. camera, expressed in camera frame. p_cam = qIm2Cam*p_image + tIm2Cam
    tIm2Cam = rotateVec(tIm2World-tCam,qConj(qCam))
    qIm2Cam = qMult(qConj(qCam),qIm2World)
    
    print tIm2Cam
    print qIm2Cam


if __name__ == '__main__':
    calibration()

