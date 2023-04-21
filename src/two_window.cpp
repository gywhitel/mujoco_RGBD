#include "mujoco_rgbd_camera.hpp"

#include <chrono>
#include <thread>
#include <mutex>

// mjUI ui;

// MuJoCo basic data structures
mjModel* model = NULL;
mjData* data = NULL;

mjvCamera camera;
mjvScene scene;
mjvOption option;

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

std::mutex mtx;

void simulation(mjModel* model, mjData* data)
{
  GLFWwindow* window = glfwCreateWindow(1200, 800, "Simulation", NULL, NULL);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  mjvPerturb perturb;
  mjrContext context;

  // Initialize visualization data
  mjv_defaultCamera(&camera);
  mjv_defaultOption(&option);
  mjv_defaultScene(&scene);
  mjr_defaultContext(&context);

  // create scene and context
  mjv_makeScene(model, &scene, 1000);
  mjr_makeContext(model, &context, mjFONTSCALE_150);

  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetCursorPosCallback(window, mouse_move);
  glfwSetScrollCallback(window, scroll);


// d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')], :][1] = next_target_depth
  auto camera_body_id = model->body_mocapid[mj_name2id(model, mjOBJ_BODY, "camera")];
  printf("Number of mocap bodies: %d\n", model->nmocap);

  int i = 0;
  while (!glfwWindowShouldClose(window))
  {
    mjtNum simstart = data->time;
    while (data->time - simstart < 1.0 / 60.0)
      mj_step(model, data); // step simulation for 1/60 s
    // get framebuffer viewport
    mjrRect viewport = {0,0,0,0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    
    // update scene and render
    mjv_updateScene(model, data, &option, NULL, &camera, mjCAT_ALL, &scene);
    mjr_render(viewport, &scene, &context);

    mjtNum* camera_pos = &(data->mocap_pos[camera_body_id]);
    camera_pos[camera_body_id*3 + 1] = sin(i/60);

    // printf("[%3f, %3f, %3f]\n", camera_pos[camera_body_id*3], camera_pos[camera_body_id*3 + 1], camera_pos[camera_body_id*3 + 2]);

    // Swap OpenGL buffers
    glfwSwapBuffers(window);

    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();
    i++;
  }

  mjv_freeScene(&scene);
  mjr_freeContext(&context);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);


void RGBD_sensor(mjModel* model, mjData* data)
{
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  GLFWwindow* window = glfwCreateWindow(1200, 800, "Camera", NULL, NULL);
  // glfwSetWindowAttrib(window, GLFW_RESIZABLE, GLFW_FALSE);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  // setup camera
  mjvCamera rgbd_camera;
  rgbd_camera.type = mjCAMERA_FIXED;
  rgbd_camera.fixedcamid = mj_name2id(model, mjOBJ_CAMERA, "camera");
  
  mjvOption sensor_option;
  mjvPerturb sensor_perturb;
  mjvScene sensor_scene;
  mjrContext sensor_context;

  mjv_defaultOption(&sensor_option);
  mjv_defaultScene(&sensor_scene);
  mjr_defaultContext(&sensor_context);

  // create scene and context
  mjv_makeScene(model, &sensor_scene, 1000);
  mjr_makeContext(model, &sensor_context, mjFONTSCALE_150);

  RGBD_mujoco mj_RGBD;

  while (!glfwWindowShouldClose(window))
  {
    // get framebuffer viewport
    mjrRect viewport = {0,0,0,0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    
    mj_RGBD.set_camera_intrinsics(model, rgbd_camera, viewport);

    // update scene and render
    mjv_updateScene(model, data, &sensor_option, NULL, &rgbd_camera, mjCAT_ALL, &sensor_scene);
    mjr_render(viewport, &sensor_scene, &sensor_context);

    mj_RGBD.get_RGBD_buffer(model, viewport, &sensor_context);

    mtx.lock();
    *color_cloud = mj_RGBD.generate_color_pointcloud();
    mtx.unlock();

    // Swap OpenGL buffers
    glfwSwapBuffers(window);

    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();

    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    // Do not forget to release buffer to avoid memory leak
    mj_RGBD.release_buffer();
  }

  mjv_freeScene(&sensor_scene);
  mjr_freeContext(&sensor_context);
}

void pointcloud_view()
{
  pcl::visualization::PCLVisualizer::Ptr viewer = color_cloud_visual(color_cloud);

  bool INIT = true;
  while(!viewer->wasStopped() )
  {
    mtx.lock();
    viewer->updatePointCloud(color_cloud, "cloud");
    mtx.unlock();
    viewer->spinOnce(1, true);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }
}

int main(int argc, char** argv)
{
  char error[1000] = "Fail to load model";
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
  
  std::thread simulation_thread(simulation, model, data);
  std::thread visual_thread(RGBD_sensor, model, data);
  std::thread pointcloud_view_thread(pointcloud_view);
  
  simulation_thread.join();
  visual_thread.join();
  pointcloud_view_thread.join();

  mj_deleteData(data);
  mj_deleteModel(model);

  return 1;
}