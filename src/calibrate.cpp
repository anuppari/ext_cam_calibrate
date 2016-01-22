#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <image_geometry/pinhole_camera_model.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

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
    States currentState_ = STATE_WAITING; // state machine
    
    // Camera Parameters
    bool gotCamParam_ = false;
    cv::Mat camMat_;
    cv::Mat distCoeffs_;
    
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
    
    // Ring buffers
    std::queue cornerBuff_; // Buffer for storing corners during sampling
    Eigen::MatrixXd tIm2Board_; // Buffer for storing PnP solution during sampling
    Eigen::MatrixXd qIm2Board_; // Buffer for storing PnP solution during sampling
    Eigen::MatrixXd tCam_; // Buffer for storing camera position w.r.t. world during sampling
    Eigen::MatrixXd qCam_; // Buffer for storing camera orientation w.r.t. world during sampling
    Eigen::MatrixXd tBoard_; // Buffer for storing board position w.r.t. world during sampling
    Eigen::MatrixXd qBoard_; // Buffer for storing board orientation w.r.t. world during sampling
    Eigen::MatrixXd tIm2Cam_; // Buffer for storing each measurement
    Eigen::MatrixXd qIm2Cam_; // Buffer for storing each measurement

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
        
        // Initialize Buffers
        tIm2Board_.resize(sampleBuffSize_,3);
        qIm2Board_.resize(sampleBuffSize_,4);
        tCam_.resize(sampleBuffSize_,3);
        qCam_.resize(sampleBuffSize_,4);
        tBoard_.resize(sampleBuffSize_,3);
        qBoard_.resize(sampleBuffSize_,4);
        tIm2Cam_.resize(numMeasurements_,3);
        qIm2Cam_.resize(numMeasurements_,4);

        // Get camera parameters
        std::cout << "Getting camera parameters on topic: "+cameraName_+"/camera_info" << std::endl;
        camInfoSub_ = nh.subscribe(cameraName_+"/camera_info",1,&SubscribeAndPublish::camInfoCB,this);
        ROS_DEBUG("Waiting for camera parameters ...");
        do {
            ros::spinOnce();
            ros::Duration(0.1).sleep();
        } while (!(ros::isShuttingDown()) and !gotCamParam_);
        ROS_DEBUG("Got camera parameters");

        // Subscribe to images
        image_sub_ = it_.subscribe(cameraName+"/image_raw", 1, &ext_cam_calibrate::imageCb,this);
		
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
        
        //unregister subscriber
        camInfoSub_.shutdown();
        gotCamParam_ = true;
    }
    
}; // end bebop_gimbal_tf


int main(int argc, char** argv)
{
    ros::init(argc, argv, "calibrate_node");
    
    ext_cam_calibrate obj();
    
    ros::spin();
    return 0;
}
