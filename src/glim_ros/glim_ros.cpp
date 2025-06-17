#include <glim_ros/glim_ros.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <ros/package.h>

#include <gtsam_points/optimizers/linearization_hook.hpp>
#include <gtsam_points/cuda/nonlinear_factor_set_gpu_create.hpp>

#include <glim/util/config.hpp>
#include <glim/util/logging.hpp>
#include <glim/util/time_keeper.hpp>
#include <glim/util/extension_module.hpp>
#include <glim/util/extension_module_ros.hpp>
#include <glim/util/ros_cloud_converter.hpp>
#include <glim/preprocess/cloud_preprocessor.hpp>
#include <glim/odometry/async_odometry_estimation.hpp>
#include <glim/odometry/odometry_estimation_ct.hpp>
#include <glim/odometry/odometry_estimation_cpu.hpp>
#include <glim/odometry/odometry_estimation_gpu.hpp>
#include <glim/mapping/async_sub_mapping.hpp>
#include <glim/mapping/sub_mapping.hpp>
#include <glim/mapping/async_global_mapping.hpp>
#include <glim/mapping/global_mapping.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace glim {

std::string GlimROS::generateTimestampFilename(const std::string& prefix, 
                                      const std::string& extension) const{
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << prefix << "_"
       << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") 
       << extension;
    
    return ss.str();
}

bool GlimROS::saveToPCD(const std::vector<Eigen::Vector4d>& points, 
                              const std::string& save_directory) const {

    auto logger = spdlog::default_logger();

    if (points.empty()) {
        logger->warn("No points to save, skipping PCD generation");
        return false;
    }
    
    logger->info("Starting PCD file generation with {%d} points", points.size());
    
    try {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
        
        cloud->points.reserve(points.size());
        cloud->width = points.size();
        cloud->height = 1;  
        cloud->is_dense = true;  
        
        #pragma omp parallel for if(points.size() > 10000)
        for (size_t i = 0; i < points.size(); ++i) {
            const auto& eigen_point = points[i];
            
            pcl::PointXYZI pcl_point;
            pcl_point.x = static_cast<float>(eigen_point.x());
            pcl_point.y = static_cast<float>(eigen_point.y());
            pcl_point.z = static_cast<float>(eigen_point.z());
            pcl_point.intensity = static_cast<float>(eigen_point.w());  
            
            if (std::isfinite(pcl_point.x) && std::isfinite(pcl_point.y) && 
                std::isfinite(pcl_point.z) && std::isfinite(pcl_point.intensity)) {
                #pragma omp critical
                {
                    cloud->points.push_back(pcl_point);
                }
            }
        }

        cloud->width = cloud->points.size();
        
        if (cloud->points.empty()) {
            logger->error("All points are invalid, cannot save PCD file");
            return false;
        }
        
        std::string filename = generateTimestampFilename("pointcloud", ".pcd");
        std::string full_path = save_directory;
        
        if (!full_path.empty() && full_path.back() != '/') {
            full_path += "/";
        }
        full_path += filename;

        boost::filesystem::path dir_path(save_directory);
        if (!boost::filesystem::exists(dir_path)) {
            if (!boost::filesystem::create_directories(dir_path)) {
                logger->error("Failed to create directory: {%s}", save_directory);
                return false;
            }
        }
        
        logger->info("Saving {%d} valid points to: {%s}", cloud->points.size(), full_path);
        
        int result = pcl::io::savePCDFile(full_path, *cloud);
        
        if (result == 0) {
            logger->info("Successfully saved PCD file: {}", full_path);
            
            boost::filesystem::path file_path(full_path);
            if (boost::filesystem::exists(file_path)) {
                auto file_size = boost::filesystem::file_size(file_path);
                logger->info("PCD file size: {%.2f} MB", static_cast<double>(file_size) / (1024.0 * 1024.0));
            }
            
            return true;
        } else {
            logger->error("Failed to save PCD file: {%s}, error code: {%d}", full_path, result);
            return false;
        }
    } catch (const std::exception& e) {
        logger->error("Exception occurred while saving PCD file: {%s}", e.what());
        return false;
    } catch (...) {
        logger->error("Unknown exception occurred while saving PCD file");
        return false;
    }
}

GlimROS::~GlimROS() {

    auto logger = spdlog::default_logger();
    try {
        logger->info("GlimROS destructor: Triggering final map save");
        
     
            auto points = global_mapping->export_points();
            if (!points.empty()) {
                std::string save_directory ("pointclouds");
                if (save_directory.empty()) {
                    save_directory = "/tmp/glim_global_map";
                }
                
                logger->info("Saving global map to: {%s}", save_directory);
                saveToPCD(points, save_directory);
            } else {
                logger->warn("No points to save in the global map");
            }
            
            logger->info("Final point cloud contains {%d} points", points.size());
        
        
    } catch (const std::exception& e) {
        logger->error("Exception in GlimROS destructor: {%s}", e.what());
    }



  kill_switch = true;
  thread.join();
}

GlimROS::GlimROS(ros::NodeHandle& nh) {
  // Setup logger
  auto logger = spdlog::default_logger();
  auto ringbuffer_sink = get_ringbuffer_sink();
  logger->sinks().push_back(ringbuffer_sink);
  glim::set_default_logger(logger);

  if (nh.param<bool>("debug", false)) {
    spdlog::info("enable debug printing");
    logger->set_level(spdlog::level::trace);

    if (!logger->sinks().empty()) {
      auto console_sink = logger->sinks()[0];
      console_sink->set_level(spdlog::level::debug);
    }

    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("/tmp/glim_log.log", true);
    file_sink->set_level(spdlog::level::trace);
    logger->sinks().push_back(file_sink);
  }

  spdlog::info("register linearization hooks");
#ifdef BUILD_GTSAM_POINTS_GPU
  gtsam_points::LinearizationHook::register_hook([]() { return gtsam_points::create_nonlinear_factor_set_gpu(); });
#endif

  std::string config_path = nh.param<std::string>("config_path", "config");
  if (config_path[0] != '/') {
    // config_path is relative to the glim directory
    config_path = ros::package::getPath("glim") + "/" + config_path;
  }

  spdlog::info("config_path: {}", config_path);
  glim::GlobalConfig::instance(config_path);
  glim::Config config_ros(glim::GlobalConfig::get_config_path("config_ros"));

  keep_raw_points = config_ros.param<bool>("glim_ros", "keep_raw_points", false);

  // Preprocessing
  imu_time_offset = config_ros.param<double>("glim_ros", "imu_time_offset", 0.0);
  acc_scale = config_ros.param<double>("glim_ros", "acc_scale", 1.0);

  time_keeper.reset(new glim::TimeKeeper);
  preprocessor.reset(new glim::CloudPreprocessor);

  // Odometry estimation
  glim::Config config_odometry(glim::GlobalConfig::get_config_path("config_odometry"));
  const std::string odometry_estimation_so_name = config_odometry.param<std::string>("odometry_estimation", "so_name", "libodometry_estimation_cpu.so");
  spdlog::info("load {}", odometry_estimation_so_name);

  std::shared_ptr<glim::OdometryEstimationBase> odom = OdometryEstimationBase::load_module(odometry_estimation_so_name);
  if (!odom) {
    spdlog::critical("failed to load odometry estimation module");
    abort();
  }
  odometry_estimation.reset(new glim::AsyncOdometryEstimation(odom, odom->requires_imu()));

  // Sub mapping
  if (config_ros.param<bool>("glim_ros", "enable_local_mapping", true)) {
    const std::string sub_mapping_so_name = glim::Config(glim::GlobalConfig::get_config_path("config_sub_mapping")).param<std::string>("sub_mapping", "so_name", "libsub_mapping.so");
    if (!sub_mapping_so_name.empty()) {
      spdlog::info("load {}", sub_mapping_so_name);
      auto sub = SubMappingBase::load_module(sub_mapping_so_name);
      if (sub) {
        sub_mapping.reset(new AsyncSubMapping(sub));
      }
    }
  }

  // Global mapping
  if (config_ros.param<bool>("glim_ros", "enable_global_mapping", true)) {
    const std::string global_mapping_so_name =
      glim::Config(glim::GlobalConfig::get_config_path("config_global_mapping")).param<std::string>("global_mapping", "so_name", "libglobal_mapping.so");
    if (!global_mapping_so_name.empty()) {
      spdlog::info("load {}", global_mapping_so_name);
      auto global = GlobalMappingBase::load_module(global_mapping_so_name);
      if (global) {
        global_mapping.reset(new AsyncGlobalMapping(global));
      }
    }
  }

  // Extention modules
  const auto extensions = config_ros.param<std::vector<std::string>>("glim_ros", "extension_modules");
  if (extensions && !extensions->empty()) {
    for (const auto& extension : *extensions) {
      if (extension.find("viewer") == std::string::npos) {
        spdlog::warn("Extension modules are enabled!!");
        spdlog::warn("You must carefully check and follow the licenses of ext modules");

        const std::string config_ext_path = ros::package::getPath("glim_ext") + "/config";
          spdlog::info("config_ext_path: {}", config_ext_path);
        glim::GlobalConfig::instance()->override_param<std::string>("global", "config_ext", config_ext_path);

        break;
      }
    }

    for (const auto& extension : *extensions) {
      spdlog::info("load {}", extension);
      auto ext_module = ExtensionModule::load_module(extension);
      if (ext_module == nullptr) {
        spdlog::error("failed to load {}", extension);
        continue;
      } else {
        extension_modules.push_back(ext_module);

        auto ext_module_ros = std::dynamic_pointer_cast<ExtensionModuleROS>(ext_module);
        if (ext_module_ros) {
          const auto subs = ext_module_ros->create_subscriptions();
          extension_subs.insert(extension_subs.end(), subs.begin(), subs.end());
        }
      }
    }
  }

  // Start process loop
  kill_switch = false;
  thread = std::thread([this] { loop(); });

  spdlog::debug("initialized");
}

// GlimROS::~GlimROS() {
//   kill_switch = true;
//   thread.join();
// }

const std::vector<std::shared_ptr<ExtensionModule>>& GlimROS::extensions() {
  return extension_modules;
}

const std::vector<std::shared_ptr<GenericTopicSubscription>>& GlimROS::extension_subscriptions() {
  return extension_subs;
}

void GlimROS::insert_image(const double stamp, const cv::Mat& image) {
  spdlog::trace("image: {:.6f}", stamp);

  odometry_estimation->insert_image(stamp, image);

  if (sub_mapping) {
    sub_mapping->insert_image(stamp, image);
  }
  if (global_mapping) {
    global_mapping->insert_image(stamp, image);
  }
}

void GlimROS::insert_imu(double stamp, const Eigen::Vector3d& linear_acc, const Eigen::Vector3d& angular_vel) {
  spdlog::trace("IMU: {:.6f}", stamp);

  stamp += imu_time_offset;
  time_keeper->validate_imu_stamp(stamp);

  odometry_estimation->insert_imu(stamp, acc_scale * linear_acc, angular_vel);

  if (sub_mapping) {
    sub_mapping->insert_imu(stamp, acc_scale * linear_acc, angular_vel);
  }
  if (global_mapping) {
    global_mapping->insert_imu(stamp, acc_scale * linear_acc, angular_vel);
  }
}

void GlimROS::insert_frame(const glim::RawPoints::Ptr& raw_points) {
  spdlog::trace("points: {:.6f}", raw_points->stamp);

  time_keeper->process(raw_points);
  // auto preprocessed = preprocessor->preprocess(raw_points->stamp, raw_points->times, raw_points->points);
  auto preprocessed = preprocessor->preprocess(raw_points);

  if (!preprocessed || preprocessed->size() < 100) {
    spdlog::warn("skipping frame with too few points");
    return;
  }

  if (keep_raw_points) {
    // note: Raw points are used only in extension modules for visualization purposes.
    //       If you need to reduce the memory footprint, you can safely comment out the following line.
    preprocessed->raw_points = raw_points;
  }

  while (odometry_estimation->workload() > 10) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  odometry_estimation->insert_frame(preprocessed);
}

void GlimROS::loop() {
  while (!kill_switch) {
    for (const auto& ext_module : extension_modules) {
      if (!ext_module->ok()) {
        ros::shutdown();
      }
    }

    std::vector<glim::EstimationFrame::ConstPtr> estimation_results;
    std::vector<glim::EstimationFrame::ConstPtr> marginalized_frames;
    odometry_estimation->get_results(estimation_results, marginalized_frames);

    if (estimation_results.empty() && marginalized_frames.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (sub_mapping) {
      for (const auto& marginalized_frame : marginalized_frames) {
        sub_mapping->insert_frame(marginalized_frame);
      }
      const auto submaps = sub_mapping->get_results();

      if (global_mapping) {
        for (const auto& submap : submaps) {
          global_mapping->insert_submap(submap);
        }
      }
    }
  }
}

void GlimROS::save(const std::string& path) {
  if (global_mapping) {
    global_mapping->save(path);
  }
}

void GlimROS::wait(bool auto_quit) {
  spdlog::info("waiting for odometry estimation");
  odometry_estimation->join();

  if (sub_mapping) {
    std::vector<glim::EstimationFrame::ConstPtr> estimation_results;
    std::vector<glim::EstimationFrame::ConstPtr> marginalized_frames;
    odometry_estimation->get_results(estimation_results, marginalized_frames);
    for (const auto& marginalized_frame : marginalized_frames) {
      sub_mapping->insert_frame(marginalized_frame);
    }

    spdlog::info("waiting for local mapping");
    sub_mapping->join();
    const auto submaps = sub_mapping->get_results();

    if (global_mapping) {
      for (const auto& submap : submaps) {
        global_mapping->insert_submap(submap);
      }

      spdlog::info("waiting for global mapping");
      global_mapping->join();
    }
  }

  if (!auto_quit) {
    bool terminate = false;
    while (ros::ok() && !terminate) {
      for (const auto& ext_module : extension_modules) {
        terminate |= (!ext_module->ok());
      }
    }
  }
}

}  // namespace glim
