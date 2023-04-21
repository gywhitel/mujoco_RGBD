// @author: Gao Yinghao
// By Xiaomi Robotics Lab
// email: gaoyinghao@xiaomi.com

#include "mujoco_rgbd_camera.hpp"

// #include <thread>

cv::Mat RGBD_mujoco::linearize_depth(const cv::Mat& depth)
{
  cv::Mat depth_img(depth.size(), CV_32F, cv::Scalar(0));

  for (uint i = 0; i < depth_img.rows; i++)
  {
    auto* raw_depth_ptr = depth.ptr<float>(i); 
    float* m_depth_ptr = depth_img.ptr<float>(i); 

    for (uint j = 0; j < depth_img.cols; j++)
    {
      m_depth_ptr[j] = z_near * z_far * extent / (z_far - raw_depth_ptr[j] * (z_far - z_near));
    }
  }
  return depth_img;
}

void RGBD_mujoco::set_camera_intrinsics(const mjModel* model, const mjvCamera camera, const mjrRect viewport)
{
  // vertical FOV 
  double fovy = model->cam_fovy[camera.fixedcamid] / 180 * M_PI / 2;
  
  // focal length, fx = fy
  f = viewport.height / 2 / tan(fovy);

  // principal points
  cx = viewport.width / 2;
  cy = viewport.height / 2;
}

void RGBD_mujoco::get_RGBD_buffer(const mjModel* model, const mjrRect viewport, const mjrContext* context)
{
  // Use preallocated buffer to fetch color buffer and depth buffer in OpenGL
  color_buffer = (uchar*) malloc(viewport.height*viewport.width * 3);
  depth_buffer = (float*) malloc(viewport.height*viewport.width * 4);
  mjr_readPixels(color_buffer, depth_buffer, viewport, context);

  extent = model->stat.extent;
  z_near = model->vis.map.znear;
  z_far = model->vis.map.zfar;

  cv::Size img_size(viewport.width, viewport.height);
  cv::Mat rgb(img_size, CV_8UC3, color_buffer);
  cv::flip(rgb, rgb, 0);
  rgb.copyTo(color_image);

  cv::Mat depth(img_size, CV_32F, depth_buffer);
  cv::flip(depth, depth, 0);
  cv::Mat depth_img_m = linearize_depth(depth);
  depth_img_m.copyTo(depth_image);
}

pcl::PointCloud<pcl::PointXYZ> RGBD_mujoco::generate_pointcloud()
{
  using namespace pcl;
  PointCloud<PointXYZ> cloud;

  for (int i = 0; i < depth_image.rows; i++)
  {
    for (int j = 0; j < depth_image.cols; j++)
    {
      double depth = *(depth_image.ptr<float>(i,j));
      // filter far points
      if (depth < z_far)
      {
        PointXYZ point;
        point.x = double(j - cx) * depth / f;
        point.y = double(i - cy) * depth / f;
        point.z = depth;

        cloud.push_back(point);
      }
    }
  }
  return cloud;
}


pcl::PointCloud<pcl::PointXYZRGB> RGBD_mujoco::generate_color_pointcloud()
{
  using namespace pcl;
  // color image and depth image should have the same size and should be aligned
  assert(color_image.size() == depth_image.size());

  PointCloud<PointXYZRGB> rgb_cloud;

  for (int i = 0; i < color_image.rows; i++)
  {
    for (int j = 0; j < color_image.cols; j++)
    {
      double depth = *(depth_image.ptr<float>(i,j));
      // filter far points
      if (depth < z_far)
      {
        PointXYZRGB rgb_3d_point;
        rgb_3d_point.x = double(j - cx) * depth / f;
        rgb_3d_point.y = double(i - cy) * depth / f;
        rgb_3d_point.z = depth;

        const uchar* bgr_ptr = color_image.ptr<uchar>(i,j);
        rgb_3d_point.r = bgr_ptr[0];
        rgb_3d_point.g = bgr_ptr[1];
        rgb_3d_point.b = bgr_ptr[2];
        rgb_cloud.push_back(rgb_3d_point);
      }
    }
  }
  return rgb_cloud;
}

pcl::visualization::PCLVisualizer::Ptr color_cloud_visual (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (255, 255, 255);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, "cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
  viewer->addCoordinateSystem (0.5);
  viewer->setSize(1280, 840);
  return (viewer);
}

pcl::visualization::PCLVisualizer::Ptr cloud_visual (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
  viewer->addCoordinateSystem (0.5);
  viewer->setSize(1280, 840);
  return (viewer);
}