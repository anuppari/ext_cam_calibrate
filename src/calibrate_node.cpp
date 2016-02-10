#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <sstream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

static const std::string OPENCV_WINDOW = "Image window";

enum States
{
    STATE_WAITING,
    STATE_SAMPLING,
    STATE_MOVING
};

class ext_cam_calibrate
{
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Subscriber camInfoSub_;
    tf::TransformListener tfl_;
    tf::TransformBroadcaster tfbr_;
    States currentState_; // state machine
    
    // Camera Parameters
    bool gotCamParam_;
    cv::Mat camMat_;
    cv::Mat distCoeffs_;
    int imgWidth_;
    int imgHeight_;
    
    // Node Parameters
    double squareSize_; // checkerboard square size in [m]
    double zOffset_; // offset between center of mocap markers and checkboard surface [m]. Negative if markers on top of checkerboard
    int numHorizCorners_; // Number of checkerboard corners in horizontal direction (number of boxes -1)
    int numVertCorners_; // Number of checkerboard corners in vertical direction (number of boxes -1)
    int sampleBuffSize_; // Number of samples to average at each position
    double stationaryThreshold_; // Camera considered stationary if last sampleBuffSize samples of corner locations have rms error less than this
    int numMeasurements_; // Number of different positions to acquire data
    std::string cameraName_; // Camera topic, i.e. calibration should be on cameraName/cam_info and images on cameraName/image_raw
    std::string cameraTF_; // Name of tf published by mocap attached to camera
    double numMeasurementsRecorded_; // Number of measurements taken so far
    double numSamplesRecorded_; // Number of samples taken for current measurement
    
    // Ring buffers
    Eigen::MatrixXd tIm2BoardBuff_; // Buffer for storing PnP solution during sampling
    Eigen::MatrixXd qIm2BoardBuff_; // Buffer for storing PnP solution during sampling
    Eigen::MatrixXd tCamBuff_; // Buffer for storing camera position w.r.t. world during sampling
    Eigen::MatrixXd qCamBuff_; // Buffer for storing camera orientation w.r.t. world during sampling
    Eigen::MatrixXd tBoardBuff_; // Buffer for storing board position w.r.t. world during sampling
    Eigen::MatrixXd qBoardBuff_; // Buffer for storing board orientation w.r.t. world during sampling
    Eigen::MatrixXd tIm2CamBuff_; // Buffer for storing each measurement
    Eigen::MatrixXd qIm2CamBuff_; // Buffer for storing each measurement
    std::vector<cv::Point3f> chessboardObjPts_; // Stores 3D coordinates of chessboard corners

public:
    ext_cam_calibrate() : it_(nh_)
    {
        // Get parameters
        ros::NodeHandle nhp("~"); // "private" nodehandle, used to access private parameters
        nhp.param<double>("squareSize", squareSize_, 0.06);
        nhp.param<double>("zOffset", zOffset_, -0.007);
        nhp.param<int>("numHorizCorners", numHorizCorners_, 8);
        nhp.param<int>("numVertCorners", numVertCorners_, 6);
        nhp.param<int>("sampleBuffSize", sampleBuffSize_, 10);
        nhp.param<double>("stationaryThreshold", stationaryThreshold_, 1);
        nhp.param<int>("numMeasurements", numMeasurements_, 5);
        nhp.param<std::string>("cameraName", cameraName_, "camera");
        nhp.param<std::string>("cameraTF", cameraTF_, "camera");
        
        // Initialize counters
        currentState_ = STATE_WAITING;
        gotCamParam_ = false;
        numMeasurementsRecorded_ = 0;
        numSamplesRecorded_ = 0;
        
        // Initialize Buffers
        tIm2BoardBuff_ = Eigen::MatrixXd::Zero(sampleBuffSize_,3);
        qIm2BoardBuff_ = Eigen::MatrixXd::Zero(sampleBuffSize_,4);
        tCamBuff_ = Eigen::MatrixXd::Zero(sampleBuffSize_,3);
        qCamBuff_ = Eigen::MatrixXd::Zero(sampleBuffSize_,4);
        tBoardBuff_ = Eigen::MatrixXd::Zero(sampleBuffSize_,3);
        qBoardBuff_ = Eigen::MatrixXd::Zero(sampleBuffSize_,4);
        tIm2CamBuff_ = Eigen::MatrixXd::Zero(numMeasurements_,3);
        qIm2CamBuff_ = Eigen::MatrixXd::Zero(numMeasurements_,4);
        
        // Construct chessboard 3D points container
        chessboardObjPts_ = std::vector<cv::Point3f>(numHorizCorners_*numVertCorners_,cv::Point3f());
        for (int i = 0; i < numVertCorners_; i++)
        {
            for (int j = 0; j < numHorizCorners_; j++)
            {
                chessboardObjPts_.at(numHorizCorners_*i + j) = cv::Point3f((j+1)*squareSize_,(i+1)*squareSize_,zOffset_);
            }
        }

        // Get camera parameters
        std::cout << "Getting camera parameters on topic: "+cameraName_+"/camera_info" << std::endl;
        camInfoSub_ = nh_.subscribe(cameraName_+"/camera_info",1,&ext_cam_calibrate::camInfoCB,this);
        ROS_DEBUG("Waiting for camera parameters ...");
        do {
            ros::spinOnce();
            ros::Duration(0.1).sleep();
        } while (!(ros::isShuttingDown()) and !gotCamParam_);
        ROS_DEBUG("Got camera parameters");

        // Subscribe to images
        image_sub_ = it_.subscribe(cameraName_+"/image_raw", 1, &ext_cam_calibrate::imageCB,this);
        
        // Create window to display image, corners, status, etc.
        cv::namedWindow(OPENCV_WINDOW);
    }
    
    ~ext_cam_calibrate()
    {
        cv::destroyWindow(OPENCV_WINDOW);
    }
    
    // callback for getting camera intrinsic parameters
    void camInfoCB(const sensor_msgs::CameraInfoConstPtr& camInfoMsg)
    {
        //get camera info
        image_geometry::PinholeCameraModel cam_model;
        cam_model.fromCameraInfo(camInfoMsg);
        camMat_ = cv::Mat(cam_model.fullIntrinsicMatrix());
        camMat_.convertTo(camMat_,CV_32FC1);
        cam_model.distortionCoeffs().convertTo(distCoeffs_,CV_32FC1);
        imgWidth_ = camInfoMsg->width;
        imgHeight_ = camInfoMsg->height;
        
        //unregister subscriber
        camInfoSub_.shutdown();
        gotCamParam_ = true;
    }
    
    // image callback
    void imageCB(const sensor_msgs::ImageConstPtr& msg)
    {
        // Image Capture timestamp
        ros::Time imageTimeStamp = msg->header.stamp;
        
        // convert to opencv image
        cv_bridge::CvImagePtr cv_ptr;
        cv::Mat image;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            image = cv_ptr->image;
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        
        // Check if chessboard is visible
        cv::Size patternSize(numHorizCorners_,numVertCorners_);
        std::vector<cv::Point2f> corners;
        bool patternFound = cv::findChessboardCorners(image,patternSize,corners,cv::CALIB_CB_ADAPTIVE_THRESH+cv::CALIB_CB_NORMALIZE_IMAGE+cv::CALIB_CB_FAST_CHECK);
        if (patternFound)
        {
            cv::cornerSubPix(image,corners,cv::Size(11,11),cv::Size(-1,-1),cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
        }
        
        // Calculate PnP solution, and publish on tf. Draw corners
        Eigen::Vector3d tIm2Board(0,0,0);
        Eigen::Quaterniond qIm2Board(1,0,0,0);
        if (patternFound)
        {
            // Get calibration board pose relative to image sensor, expressed in image frame. p_image = rvec*p_board + tvec
            cv::Mat rvec;
            cv::Mat tvec;
            cv::solvePnP(chessboardObjPts_,corners,camMat_,distCoeffs_,rvec,tvec);
            
            // Calculate image pose relative to calibration board, expressed in board frame. p_board = qIm2Board*p_image + tIm2Board
            cv::Mat cvRotMat;
            cv::Rodrigues(rvec,cvRotMat);
            Eigen::Matrix3d rotMat;
            cv::cv2eigen(cvRotMat,rotMat);
            qIm2Board = Eigen::Quaterniond(rotMat.transpose());
            Eigen::Vector3d tBoard2Im;
            cv::cv2eigen(tvec,tBoard2Im);
            tIm2Board = -1*qIm2Board*tBoard2Im;
            
            // Broadcast for visual verification of PnP solution
            tfbr_.sendTransform(tf::StampedTransform(tf::Transform(tf::Quaternion(qIm2Board.x(),qIm2Board.y(),qIm2Board.z(),qIm2Board.w()),tf::Vector3(tIm2Board(0),tIm2Board(1),tIm2Board(2))),imageTimeStamp,"board","image2"));
            
            // Draw corners on image
            drawChessboardCorners(image, patternSize, cv::Mat(corners), patternFound);
        }
        
        // Status/info
        cv::Mat infoImg = get_status_image(image.type());
        
        // Display
        cv::Mat displayImage(image.rows+infoImg.rows, image.cols, image.type());
        image.copyTo(displayImage(cv::Range(0,image.rows),cv::Range(0,image.cols)));
        infoImg.copyTo(displayImage(cv::Range(image.rows+1,image.rows+infoImg.rows),cv::Range(0,image.cols)));
        cv::imshow(OPENCV_WINDOW,displayImage);
        int key = cv::waitKey(30); // allow time to display image and get key presses
        
        if (currentState_ == STATE_WAITING)
        {
            // check for keypresses
            if (key > 0)
            {
                if (key == 32) // space bar = start sampling
                {
                    currentState_ = STATE_SAMPLING;
                    numSamplesRecorded_ = 0;
                }
                else if (key == 13) // enter key = finished with measurements
                {
                    image_sub_.shutdown();
                    
                    // Final calibration results, and variances
                    Eigen::Vector3d t(0,0,0);
                    Eigen::Vector4d q(0,0,0,0);
                    Eigen::Vector3d t_var(0,0,0);
                    Eigen::Vector4d q_var(0,0,0,0);
                    get_current_calibration_results(t,q,t_var,q_var);
                    
                    // Print
                    std::ostringstream convert;
                    convert << "Current Calibration:" << std::endl;
                    convert << "  translation: " << t << std::endl;
                    convert << "  orientation: " << q << std::endl;
                    convert << "Measurement Variance:" << std::endl;
                    convert << "  translation: " << t_var << std::endl;
                    convert << "  orientation: " << q_var << std::endl;
                    ROS_INFO("%s",convert.str().c_str());
                    
                    // save to file
                }
            }
        }
        else if (currentState_ == STATE_SAMPLING)
        {
            if (patternFound)
            {
                // Store calculated PnP solution in buffer
                tIm2BoardBuff_.row(numSamplesRecorded_) = tIm2Board;
                qIm2BoardBuff_.row(numSamplesRecorded_) << qIm2Board.w(), qIm2Board.x(), qIm2Board.y(), qIm2Board.z(); // Eigen order, w,x,y,z
                
                // Get calibration board pose w.r.t. world, expressed in world frame. p_world = qBoard*p_board + tBoard
                tf::StampedTransform boardTransform;
                tfl_.waitForTransform("/world","/board",imageTimeStamp,ros::Duration(0.1));
                tfl_.lookupTransform("/world","/board",imageTimeStamp,boardTransform);
                tBoardBuff_.row(numSamplesRecorded_) << boardTransform.getOrigin().getX(),boardTransform.getOrigin().getY(),boardTransform.getOrigin().getZ();
                qBoardBuff_.row(numSamplesRecorded_) << boardTransform.getRotation().getW(), boardTransform.getRotation().getX(), boardTransform.getRotation().getY(), boardTransform.getRotation().getZ(); // Eigen order, w,x,y,z
                
                // Get camera pose w.r.t. world, expressed in world frame. p_world = qCam*p_cam + tCam
                tf::StampedTransform camTransform;
                tfl_.waitForTransform("/world",cameraTF_,imageTimeStamp,ros::Duration(0.1));
                tfl_.lookupTransform("/world",cameraTF_,imageTimeStamp,camTransform);
                tCamBuff_.row(numSamplesRecorded_) << camTransform.getOrigin().getX(),camTransform.getOrigin().getY(),camTransform.getOrigin().getZ();
                qCamBuff_.row(numSamplesRecorded_) << camTransform.getRotation().getW(), camTransform.getRotation().getX(), camTransform.getRotation().getY(), camTransform.getRotation().getZ(); // Eigen order, w,x,y,z
                
                if (numSamplesRecorded_ >= sampleBuffSize_)
                {
                    // Get sample mean
                    tIm2Board = tIm2BoardBuff_.colwise().mean();
                    qIm2Board = Eigen::Quaterniond((Eigen::Vector4d) qIm2BoardBuff_.colwise().mean());
                    Eigen::Vector3d tBoard = tBoardBuff_.colwise().mean();
                    Eigen::Quaterniond qBoard((Eigen::Vector4d) qBoardBuff_.colwise().mean());
                    Eigen::Vector3d tCam = tCamBuff_.colwise().mean();
                    Eigen::Quaterniond qCam((Eigen::Vector4d) qCamBuff_.colwise().mean());
                    
                    // Calculate image pose w.r.t. world, expressed in world frame. p_world = qIm2World*p_image + tIm2World
                    Eigen::Vector3d tIm2World = tBoard + qBoard*tIm2Board;
                    Eigen::Quaterniond qIm2World = qBoard*qIm2Board;
                    tfbr_.sendTransform(tf::StampedTransform(tf::Transform(tf::Quaternion(qIm2World.x(),qIm2World.y(),qIm2World.z(),qIm2World.w()),tf::Vector3(tIm2World(0),tIm2World(1),tIm2World(2))),imageTimeStamp,"board","image"));
                    
                    // Calculate image pose w.r.t. camera, expressed in camera frame. p_cam = qIm2Cam*p_image + tIm2Cam
                    Eigen::Vector3d tIm2Cam = qCam.inverse()*(tIm2World-tCam);
                    Eigen::Quaterniond qIm2Cam = qCam.inverse()*qIm2World;
                    
                    // Deal with equivalent representations
                    if (qIm2Cam.w() < 0)
                    {
                        qIm2Cam.w() = -1*qIm2Cam.w();
                        qIm2Cam.vec() = -1*qIm2Cam.vec();
                    }
                    
                    // Add to solution buffer
                    tIm2CamBuff_.row(numMeasurementsRecorded_) = tIm2Cam;
                    qIm2CamBuff_.row(numMeasurementsRecorded_) << qIm2Cam.vec(), qIm2Cam.w(); // tf order, x,y,z,w
                    
                    // Clear sample buffers
                    tIm2BoardBuff_ = Eigen::MatrixXd::Zero(sampleBuffSize_,3);
                    qIm2BoardBuff_ = Eigen::MatrixXd::Zero(sampleBuffSize_,4);
                    tCamBuff_ = Eigen::MatrixXd::Zero(sampleBuffSize_,3);
                    qCamBuff_ = Eigen::MatrixXd::Zero(sampleBuffSize_,4);
                    tBoardBuff_ = Eigen::MatrixXd::Zero(sampleBuffSize_,3);
                    qBoardBuff_ = Eigen::MatrixXd::Zero(sampleBuffSize_,4);
                    
                    numSamplesRecorded_ = 0;
                    numMeasurementsRecorded_++;
                }
                else
                {
                    numSamplesRecorded_++;
                }
            }
        }
    }
    
    // Draws image with status info
    cv::Mat get_status_image(int type)
    {
        cv::Mat infoImg(imgHeight_/4,imgWidth_,type);
        infoImg = cv::Scalar(255,255,255); // set background as white
        
        // Current calibration results, and variances
        Eigen::Vector3d t(0,0,0);
        Eigen::Vector4d q(0,0,0,0);
        Eigen::Vector3d t_var(0,0,0);
        Eigen::Vector4d q_var(0,0,0,0);
        get_current_calibration_results(t,q,t_var,q_var);
        
        // convert to string
        std::ostringstream convert;
        convert << "Num. Recorded Measurements: " << numMeasurementsRecorded_ << std::endl;
        convert << "Current Calibration:" << std::endl;
        convert << "  translation: " << t << std::endl;
        convert << "  orientation: " << q << std::endl;
        convert << "Measurement Variance:" << std::endl;
        convert << "  translation: " << t_var << std::endl;
        convert << "  orientation: " << q_var << std::endl;
        
        cv::putText(infoImg,convert.str(),cv::Point(5,10),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(0,0,0),2);
        
        return infoImg;
    }
    
    void get_current_calibration_results(Eigen::Vector3d &t,Eigen::Vector4d &q,Eigen::Vector3d &t_var,Eigen::Vector4d &q_var)
    {
        if (numMeasurementsRecorded_ > 0)
        {
            t = tIm2CamBuff_.topRows(numMeasurementsRecorded_).colwise().mean();
            q = qIm2CamBuff_.topRows(numMeasurementsRecorded_).colwise().mean();
            t_var = (tIm2CamBuff_.topRows(numMeasurementsRecorded_).rowwise() - t.transpose()).array().pow(2).colwise().sum()/numMeasurementsRecorded_;
            q_var = (qIm2CamBuff_.topRows(numMeasurementsRecorded_).rowwise() - q.transpose()).array().pow(2).colwise().sum()/numMeasurementsRecorded_;
        }
    }
    
}; // end ext_cam_calibrate


int main(int argc, char** argv)
{
    ros::init(argc, argv, "calibrate_node");
    
    ext_cam_calibrate obj();
    
    ros::spin();
    
    ROS_INFO("Finished Calibration");
    ROS_INFO("Calibration parameters written to file: ");
    return 0;
}
