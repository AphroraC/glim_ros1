#pragma once

#include <mutex>
#include <atomic>
#include <thread>

#include <ros/ros.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>

#include <glim/odometry/estimation_frame.hpp>
#include <glim/mapping/sub_map.hpp>
#include <glim/util/extension_module.hpp>

namespace spdlog {
class logger;
}

namespace glim {

class TrajectoryManager;

class RvizViewer : public ExtensionModule {
public:
  RvizViewer();
  ~RvizViewer();

private:
  void set_callbacks();
  void odometry_new_frame(const EstimationFrame::ConstPtr& new_frame);
  void process_transform_matrix(const EstimationFrame::ConstPtr& new_frame);
  void publish_aligned_points_msg(const EstimationFrame::ConstPtr& new_frame);
  void publish_transform_map2odom(const EstimationFrame::ConstPtr& new_frame);
  void publish_transform_odom2base(const EstimationFrame::ConstPtr& new_frame);
  void publish_transform_imu2lidar(const EstimationFrame::ConstPtr& new_frame);
  void publish_odometry_msg(const EstimationFrame::ConstPtr& new_frame);
  // void publish_pose_msg(const EstimationFrame::ConstPtr& new_frame);
  void globalmap_on_update_submaps(const std::vector<SubMap::Ptr>& submaps);
  void invoke(const std::function<void()>& task);
  void spin_once();

private:
  std::atomic_bool kill_switch;
  std::thread thread;

  ros::NodeHandle nh;
  ros::NodeHandle private_nh;

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener;
  tf2_ros::TransformBroadcaster tf_broadcaster;

  ros::WallTime last_globalmap_pub_time;

  std::string imu_frame_id;
  std::string lidar_frame_id;
  std::string base_frame_id;
  std::string odom_frame_id;
  std::string map_frame_id;
  bool publish_aligned_points;
  bool publish_odometry;
  bool publish_transform;
  bool publish_map2odom;
  bool publish_odom2base;
  bool publish_imu2lidar;
  double tf_time_offset;

  ros::Publisher points_pub;
  ros::Publisher map_pub;

  ros::Publisher odom_pub;
  ros::Publisher pose_pub;
  ros::Publisher transform_pub;

  std::mutex trajectory_mutex;
  std::unique_ptr<TrajectoryManager> trajectory;

  std::vector<gtsam_points::PointCloud::ConstPtr> submaps;

  std::mutex invoke_queue_mutex;
  std::vector<std::function<void()>> invoke_queue;

  // Logging
  std::shared_ptr<spdlog::logger> logger;

  // Transforms
  ros::Time stamp;
  ros::Time tf_stamp;
  geometry_msgs::TransformStamped trans;

  Eigen::Isometry3d T_odom_imu;
  Eigen::Quaterniond quat_odom_imu;

  Eigen::Isometry3d T_lidar_imu;
  Eigen::Quaterniond quat_lidar_imu;

  Eigen::Isometry3d T_world_odom;
  Eigen::Quaterniond quat_world_odom;

  Eigen::Isometry3d T_world_imu;
  Eigen::Quaterniond quat_world_imu;

  Eigen::Isometry3d T_imu_base;
  Eigen::Isometry3d T_odom_base;
  Eigen::Quaterniond quat_odom_base;
};

}  // namespace glim