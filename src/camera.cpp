// Code reference on basic.cc

// Basic mujoco visualization without mouse and keyboard event support
// Only for RGBD camera test

#include "mujoco.h"
#include "glfw3.h"
#include "cstdio"

#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <thread>

#define DEPTH_MAX 10 // maximal depth 10 m

// MuJoCo basic data structures
mjModel* model = NULL;
mjData* data = NULL;
mjvOption option;
mjvPerturb perturb;
mjvCamera camera;
int cat_mask;
mjvScene scene;
mjrContext context;

bool mouse_button_L = false;  // left mouse button
bool mouse_button_R = false;  // right mouse button
bool mouse_button_M = false;  // middle mouse button

double lastX = 0, lastY = 0;  // cursor position in last time

// mouse middle roller scroll
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
  mjv_moveCamera(model, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scene, &camera);
}

// mouse button click
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
  mouse_button_L = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
  mouse_button_R = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
  mouse_button_M = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);

  glfwGetCursorPos(window, &lastX, &lastY);
}

// mouse cursor move
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
  if ( !mouse_button_L && !mouse_button_M && !mouse_button_R)
    return;
  
  // mouse displacement
  double dx = xpos - lastX;
  double dy = ypos - lastY;
  lastX = xpos;
  lastY = ypos;

  // get window size
  int width, height;
  glfwGetWindowSize(window, &width, &height);

  // get shift key state
  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || 
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

  mjtMouse action;
  if (mouse_button_R)
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  else if (mouse_button_L)
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  else
    action = mjMOUSE_ZOOM;
  
  // move camera
  mjv_moveCamera(model, action, dx/height, dy/height, &scene, &camera);
}

/// @brief Convert depth image from MuJoCo render to depth image in meters
/// @param depth CV_32F mat
/// @return depth image in meters (CV_32F)
cv::Mat depth_to_meters(const cv::Mat& depth)
{
  // Invert depth image about X axis, because OpenGL render image in Y-flip way
  cv::flip(depth, depth, 0);
  double extent = model->stat.extent;
  double z_near = model->vis.map.znear;
  double z_far = model->vis.map.zfar;
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

/// @brief Synthesize RGB point cloud from color image and depth image
/// @param color_img cv::Mat 8UC3 (RGB order)
/// @param depth_img_m  cv::Mat 16F (in meters) 
/// @param f focal length(m)
/// @param cx principal point x of image (pixel)  
/// @param cy principal point y of image (pixel)
/// @return RGB 3D point cloud
pcl::PointCloud<pcl::PointXYZRGB> generate_RGB_pointcloud(const cv::Mat& color_img, const cv::Mat& depth_img_m, double f, int cx, int cy)
{
  using namespace pcl;
  // color image and depth image should have the same size and should be aligned
  assert(color_img.size() == depth_img_m.size());

  PointCloud<PointXYZRGB> rgb_cloud;
  double z_far = model->vis.map.zfar;
  // printf("f:%5f, cx:%d, cy:%d z_far:%3f extent:%5f\n", f, cx, cy, z_far, extent);

  for (int i = 0; i < color_img.rows; i++)
  {
    for (int j = 0; j < color_img.cols; j++)
    {
      double depth = *(depth_img_m.ptr<float>(i,j));
      // filter far points
      if (depth < z_far)
      {
        PointXYZRGB rgb_3d_point;
        rgb_3d_point.x = double(j - cx) * depth / f;
        rgb_3d_point.y = double(i - cy) * depth / f;
        rgb_3d_point.z = depth;

        const uchar* bgr_ptr = color_img.ptr<uchar>(i,j);
        rgb_3d_point.r = bgr_ptr[0];
        rgb_3d_point.g = bgr_ptr[1];
        rgb_3d_point.b = bgr_ptr[2];
        rgb_cloud.push_back(rgb_3d_point);
      }
    }
  }
  return rgb_cloud;
}


pcl::PointCloud<pcl::PointXYZ> generate_pointcloud(const cv::Mat& depth_img_m, double f, int cx, int cy)
{
  using namespace pcl;
  // color image and depth image should have the same size and should be aligned

  PointCloud<PointXYZ> cloud;
  double z_far = model->vis.map.zfar;
  // printf("f:%5f, cx:%d, cy:%d z_far:%3f extent:%5f\n", f, cx, cy, z_far, extent);

  for (int i = 0; i < depth_img_m.rows; i++)
  {
    for (int j = 0; j < depth_img_m.cols; j++)
    {
      double depth = *(depth_img_m.ptr<float>(i,j));
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

int main(int argc, char** argv)
{
  char error[1000] = "Fail to load model";

  // if (argc != 2)
  // {
  //   printf("USAGE: %s model_file\n", argv[0]);
  //   return EXIT_FAILURE;
  // }

  // Load model into mujoco 
  // DO not use ~ in path
  const char* file = "/home/yinghao/.mujoco/mujoco210/camera_test/data/camera_test.xml";
  model = mj_loadXML(file, 0, error, 1000);
  
  if (!model)
  {
    printf("%s\n", error);
    mju_error("Load model error.\n");
  }
  else
    printf("Model loaded.\n");
  
  data = mj_makeData(model);

  // init GLFW (Graphic Library FrameWork)
  if (!glfwInit())
    mju_error("Fail to initiate GLFW");
  else
    printf("GLFW initiated. \n");

  // Create GLFW window
  GLFWwindow* window = glfwCreateWindow(1200, 800, "Camera_test", NULL, NULL);
  
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  // Initialize visualization data
  mjv_defaultCamera(&camera);
  mjv_defaultOption(&option);
  mjv_defaultScene(&scene);
  mjr_defaultContext(&context);

  // create scene and context
  mjv_makeScene(model, &scene, 1000);
  mjr_makeContext(model, &context, mjFONTSCALE_150);

  // setup camera
  mjvCamera rgbd_camera;
  rgbd_camera.type = mjCAMERA_FIXED;
  rgbd_camera.fixedcamid = mj_name2id(model, mjOBJ_CAMERA, "camera");
  // mjv_defaultCamera(&rgbd_camera);  // do not set this camera as fixed camera

  mjvScene scene2;
  mjrContext context2;

  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetCursorPosCallback(window, mouse_move);
  glfwSetScrollCallback(window, scroll);

  cv::String win = "OpenCV Image";
  cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

  bool INIT = true;

  
  double fovy, f;
  uint cx, cy;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::visualization::PCLVisualizer::Ptr viewer;

  while (!glfwWindowShouldClose(window))
  {
    mjtNum simstart = data->time;
    while (data->time - simstart < 1.0 / 60.0)
      mj_step(model, data); // step simulation for 1/60 s
    // get framebuffer viewport
    mjrRect viewport = {0,0,0,0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    
    // update scene and render
    mjv_updateScene(model, data, &option, NULL, &rgbd_camera, mjCAT_ALL, &scene);
    mjr_render(viewport, &scene, &context);

    cv::Size img_size(viewport.width, viewport.height);
    
    // Pointers must be preallocated
    uchar* rgb_data = (uchar*) malloc(viewport.height*viewport.width * 3);
    float* depth_data = (float*) malloc(viewport.height*viewport.width * 4);
    mjr_readPixels(rgb_data, depth_data, viewport, &context);
    
    cv::Mat rgb(img_size, CV_8UC3, rgb_data);
    cv::flip(rgb, rgb, 0);
    // cv::Mat color_image = rgb.clone();
    // Invert image about X axis
    // // visualize color image
    // cv::cvtColor(rgb, rgb, cv::COLOR_RGB2BGR);
    // imshow(win, rgb);
    // cv::waitKey(1);

    // // convert depth matrix from [0,1] to {0,1...,255} for OpenCV visualization
    // convert depth image from [0, 1] -> distance in meters
    cv::Mat depth(img_size, CV_32F, depth_data);
    cv::Mat depth_img_m = depth_to_meters(depth);
    // cv::Mat depth_image = depth_img_m.clone();
    // // visualize depth image
    // depth_img_m = depth_img_m*255;
    // cv::Mat depth_img_visual;   // Mat for visualization
    // depth_img_m.convertTo(depth_img_visual, CV_8U);
    // imshow(win, depth_img_visual);
    // cv::waitKey(1);
    

    // another thread for visualization
    //// camera intrinsics
    // vertical FOV 
    fovy = model->cam_fovy[rgbd_camera.fixedcamid] / 180 * M_PI / 2;
    // focal length, fx = fy
    // printf("fovy: %5f ", fovy);
    f = viewport.height / 2 / tan(fovy);
    cx = viewport.width / 2;
    cy = viewport.height / 2;

    // visualize RGB point cloud
    *color_cloud = generate_RGB_pointcloud(rgb, depth_img_m, f, cx, cy);
    
    // monochrome pointcloud
    // *cloud = generate_pointcloud(depth_img_m, f, cx, cy);
     
    if (INIT)
    {
      viewer = color_cloud_visual(color_cloud);
      // viewer = cloud_visual(cloud);
      
      // DEBUG error in the visualization thread
      // std::thread cloud_visual_thread([](pcl::visualization::PCLVisualizer::Ptr viewer){
      //   // spin() returns when visualizer window is closed
      //   viewer->spin();
      // }, viewer);

      // // NOTE join the thread works well, but detaching it causes error of vtk
      // cloud_visual_thread.detach();
      
      INIT = false;
    }
    viewer->spinOnce(1);
    // viewer->updatePointCloud(color_cloud, "cloud");
    // else
    // {
    // // // Spin blocks main thread. Seperate spin in another thread
    // }

    // Swap OpenGL buffers
    glfwSwapBuffers(window);

    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();

    // free memory
    free(rgb_data);
    free(depth_data);
  }

  mjv_freeScene(&scene);
  mjr_freeContext(&context);

  mj_deleteData(data);
  mj_deleteModel(model);

  cv::destroyAllWindows();
  // terminate GLFW (crashes with Linux NVidia drivers)
  #if defined(__APPLE__) || defined(_WIN32)
      glfwTerminate();
  #endif

  return 1;

}
