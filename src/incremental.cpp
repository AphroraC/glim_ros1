// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

/**
 * @file  advanced_scan_matching.cpp
 * @brief This example demonstrates how to use iVox structure to efficiently do frame-to-map scan matching.
 */

#include <boost/format.hpp>

#include <gtsam_ext/util/read_points.hpp>
#include <gtsam_ext/util/covariance_estimation.hpp>
#include <gtsam_ext/ann/ivox.hpp>
#include <gtsam_ext/types/frame_cpu.hpp>
#include <gtsam_ext/factors/integrated_gicp_factor.hpp>
#include <gtsam_ext/factors/integrated_vgicp_factor.hpp>
#include <gtsam_ext/optimizers/levenberg_marquardt_ext.hpp>

#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <glk/normal_distributions.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: advanced_scan_matching /path/to/your/kitti/00/velodyne" << std::endl;
    return 0;
  }

  const std::string seq_path = argv[1];

  // Mapping parameters
  const int num_threads = 4;
  const double voxel_resolution = 1.0;
  const double randomsampling_rate = 0.1;

  std::mt19937 mt;
  auto viewer = guik::LightViewer::instance();
  viewer->disable_vsync();
  viewer->show_info_window();

  // Create iVox
  gtsam_ext::iVox::Ptr ivox(new gtsam_ext::iVox(voxel_resolution));
  gtsam_ext::GaussianVoxelMapCPU::Ptr voxelmap(new gtsam_ext::GaussianVoxelMapCPU(1.0));

  // Estimated sensor pose
  gtsam::Pose3 T_world_lidar;

  for (int i = 0;; i++) {
    // Read points and replace the last element (w) with 1 for homogeneous transformation
    const std::string points_path = (boost::format("%s/%06d.bin") % seq_path % i).str();
    auto points = gtsam_ext::read_points4(points_path);
    if (points.empty()) {
      break;
    }
    std::for_each(points.begin(), points.end(), [](Eigen::Vector4f& p) { p.w() = 1.0f; });

    // Create a frame and do random sampling and covariance estimation
    auto frame = std::make_shared<gtsam_ext::FrameCPU>(points);
    frame = gtsam_ext::random_sampling(frame, randomsampling_rate, mt);
    frame->add_covs(gtsam_ext::estimate_covariances(frame->points, frame->size(), 10, num_threads));

    // If it is not the first frame, do frame-to-map scan matching
    if (i != 0) {
      gtsam::Values values;
      values.insert(0, gtsam::Pose3());  // Target pose = Map origin
      values.insert(1, T_world_lidar);   // Source pose initial guess = Last sensor pose

      gtsam::NonlinearFactorGraph graph;
      // Fix the target pose
      graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));

      // Create an ICP factor between target (iVox) and source (current frame)
      // auto icp_factor = gtsam::make_shared<gtsam_ext::IntegratedGICPFactor_<gtsam_ext::iVox, gtsam_ext::Frame>>(0, 1, ivox, frame, ivox);
      auto icp_factor = gtsam::make_shared<gtsam_ext::IntegratedVGICPFactor>(0, 1, voxelmap, frame);

      icp_factor->set_num_threads(num_threads);
      graph.add(icp_factor);

      // Optimize
      gtsam_ext::LevenbergMarquardtExtParams lm_params;
      lm_params.setMaxIterations(20);
      lm_params.set_verbose();
      gtsam_ext::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
      values = optimizer.optimize();

      // Update the current pose
      T_world_lidar = values.at<gtsam::Pose3>(1);

      Eigen::Matrix4d R = Eigen::Matrix4d::Zero();
      R.block<3, 3>(0, 0) = T_world_lidar.rotation().matrix();

      // Transform the current frame into the map frame
      for (int j = 0; j < frame->size(); j++) {
        frame->points[j] = T_world_lidar.matrix() * frame->points[j];
        frame->covs[j] = R * frame->covs[j] * R.transpose();
      }
    }

    // Insert the transformed current frame into iVox
    // ivox->insert(*frame);
    voxelmap->insert(*frame);

    std::vector<Eigen::Vector3d> means;
    std::vector<Eigen::Matrix3d> covs;
    for (const auto& voxel : voxelmap->voxels) {
      means.push_back(voxel.second->mean.head<3>());
      covs.push_back(voxel.second->cov.block<3, 3>(0, 0));
    }
    viewer->update_drawable("covs", std::make_shared<glk::NormalDistributions>(means, covs, 1.0f), guik::Rainbow());

    // Visualization
    viewer->update_drawable(guik::anon(), glk::Primitives::coordinate_system(), guik::VertexColor(T_world_lidar.matrix().cast<float>()));
    viewer->update_drawable("current", std::make_shared<glk::PointCloudBuffer>(points), guik::FlatOrange(T_world_lidar.matrix().cast<float>()).add("point_scale", 2.0f));
    // viewer->update_drawable("ivox", std::make_shared<glk::PointCloudBuffer>(ivox->voxel_points()), guik::Rainbow());
    if (!viewer->spin_once()) {
      break;
    }
  }

  return 0;
}