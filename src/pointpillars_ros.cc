// headers in STL
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
// ros
#include "ros/time.h"
#include "ros/ros.h"
#include "jsk_recognition_msgs/BoundingBoxArray.h"
#include "jsk_recognition_msgs/BoundingBox.h"
#include <sensor_msgs/PointCloud2.h>
#include "pcl_conversions/pcl_conversions.h"

// pcl
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

// headers in 3rd-part
#include "../pointpillars/pointpillars.h"
#include "gtest/gtest.h"
using namespace std;

float *points_array;
int in_num_points;
float g_score_threshold;
std::string g_frame_id;
// 距离阈值
float distance_threshold = 1.5;

struct Track
{
    int id;
    jsk_recognition_msgs::BoundingBox bbox;
    int age;
    int lost;
};

std::vector<Track> tracks;
int global_track_id = 0;

const std::vector<std::string> CLASS_NAMES = {
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"};

std::string get_current_time()
{
    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);

    // 格式化为字符串
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

int Txt2Arrary(float *&points_array, string file_name, int num_feature = 4)
{
    ifstream InFile;
    InFile.open(file_name.data());
    assert(InFile.is_open());

    vector<float> temp_points;
    string c;

    while (!InFile.eof())
    {
        InFile >> c;

        temp_points.push_back(atof(c.c_str()));
    }
    points_array = new float[temp_points.size()];
    for (int i = 0; i < temp_points.size(); ++i)
    {
        points_array[i] = temp_points[i];
    }

    InFile.close();
    return temp_points.size() / num_feature;
    // printf("Done");
};

void Boxes2Txt(std::vector<float> boxes, std::vector<int> labels, string file_name, int num_feature = 7)
{
    ofstream ofFile;
    ofFile.open(file_name, std::ios::out | std::ios::app);
    if (ofFile.is_open())
    {
        for (int i = 0; i < boxes.size() / num_feature; ++i)
        {
            if (i < labels.size())
            {
                ofFile << CLASS_NAMES[labels[i]] << " ";
            }
            for (int j = 0; j < num_feature; ++j)
            {
                ofFile << boxes.at(i * num_feature + j) << " ";
            }
            ofFile << "\n";
        }
    }
    std::string time_str = get_current_time();
    ofFile << "---------------------------------" << time_str << "----------------------------" << "\n";
    ofFile.close();
    return;
};
float centerDistance(const jsk_recognition_msgs::BoundingBox &a,
                     const jsk_recognition_msgs::BoundingBox &b)
{
    float dx = a.pose.position.x - b.pose.position.x;
    float dy = a.pose.position.y - b.pose.position.y;
    float dz = a.pose.position.z - b.pose.position.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void updateTracks(const jsk_recognition_msgs::BoundingBoxArray &detections)
{
    std::vector<bool> matched(detections.boxes.size(), false);
    for (auto &track : tracks)
    {
        float min_dist = 9999;
        int best_idx = -1;
        for (size_t i = 0; i < detections.boxes.size(); i++)
        {
            float dist = centerDistance(track.bbox, detections.boxes[i]);
            if (dist < min_dist && dist < distance_threshold)
            {
                min_dist = dist;
                best_idx = i;
            }
        }
        if (best_idx != -1 && !matched[best_idx])
        {
            // 更新现有轨迹
            track.bbox = detections.boxes[best_idx];
            track.age++;
            track.lost = 0;
            matched[best_idx] = true;
        }
        else
        {
            track.lost++;
        }
    }
    // 添加未匹配的新目标
    for (size_t i = 0; i < detections.boxes.size(); i++)
    {
        if (!matched[i])
        {
            Track new_track;
            new_track.id = global_track_id++;
            new_track.bbox = detections.boxes[i];
            new_track.age = 1;
            new_track.lost = 0;
            tracks.push_back(new_track);
        }
    }
    // 移除lost太久的轨迹
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                [](const Track &track)
                                { return track.lost > 5; }),
                 tracks.end());
}
void publicTrackedBoxes(ros::Publisher &pub_bbox)
{
    jsk_recognition_msgs::BoundingBoxArray tracked_array;
    tracked_array.header.frame_id = g_frame_id;
    tracked_array.header.stamp = ros::Time::now();
    for(const auto& t : tracks){
        jsk_recognition_msgs::BoundingBox box = t.bbox;
        box.label = t.id;
        tracked_array.boxes.push_back(box);
    }
    pub_bbox.publish(tracked_array);
}

void lidar_callback(const sensor_msgs::PointCloud2ConstPtr &pc_msg_ptr)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*pc_msg_ptr, *cloud);

    auto trans_cloudxyz = cloud->getMatrixXfMap(3, 8, 0);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> trans_cloudi = cloud->getMatrixXfMap(1, 8, 4);
    Eigen::MatrixXf pointsmap(trans_cloudxyz.rows() + trans_cloudi.rows(), trans_cloudxyz.cols());
    // std::cout << "trans_cloudxyz.rows(): " << trans_cloudxyz.rows() << ", trans_cloudi.rows():" << trans_cloudi.rows()<< std::endl;
    pointsmap << trans_cloudxyz,
        trans_cloudi;
    Eigen::Matrix<float, Eigen::Dynamic, 4> data_in = pointsmap.transpose();
    std::cout << "lidar point size  " << data_in.rows() << std::endl;
    in_num_points = data_in.rows();
    if (in_num_points < 1000)
        return;

    points_array = new float[in_num_points * 5];
    for (int i = 0; i < in_num_points; i++)
    {
        points_array[i*5 + 0] = data_in(i, 0); // x, y, z,i, 0
        points_array[i*5 + 1] = data_in(i, 1);
        points_array[i*5 + 2] = data_in(i, 2);
        points_array[i*5 + 3] = data_in(i, 3);
        points_array[i*5 + 4] = 0.0;
    }
}

void publishDetectionResult(const std::vector<float> &boxes,
                            const std::vector<int> &out_labels,
                            const std::vector<float> &out_scores,
                            ros::Publisher &pub_bbox,
                            const bool &save_result,
                            const std::string &output_file)
{
    int box_size = boxes.size() / 7;
    std::cout << "inference boxes size  " << box_size << std::endl;
    if (box_size == 0)
        return;
    assert(out_labels.size() == out_scores.size());
    assert(box_size = out_labels.size());

    jsk_recognition_msgs::BoundingBoxArray jsk_boxes;
    // box_dim： x，y，z，dx，dy，dz，yaw
    for (int i = 0; i < boxes.size(); i = i + 7)
    {
        jsk_recognition_msgs::BoundingBox jsk_box;
        jsk_box.header.frame_id = g_frame_id;
        jsk_box.pose.position.x = boxes[i + 0];
        jsk_box.pose.position.y = boxes[i + 1];
        jsk_box.pose.position.z = boxes[i + 2];
        jsk_box.dimensions.x = boxes[i + 3];
        jsk_box.dimensions.y = boxes[i + 4];
        jsk_box.dimensions.z = boxes[i + 5];
        // yaw
        Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()));
        Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()));
        Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(boxes[i + 6], Eigen::Vector3d::UnitZ()));
        Eigen::Quaterniond quaternion;
        quaternion = yawAngle * pitchAngle * rollAngle;
        jsk_box.pose.orientation.w = quaternion.w();
        jsk_box.pose.orientation.x = quaternion.x();
        jsk_box.pose.orientation.y = quaternion.y();
        jsk_box.pose.orientation.z = quaternion.z();
        int box_idx = i / 7;
        jsk_box.label = out_labels[box_idx];
        jsk_box.value = out_scores[box_idx];
        // 利用分数过滤
        if (jsk_box.value > g_score_threshold)
        {
            jsk_boxes.boxes.emplace_back(jsk_box);
        }
    }
    jsk_boxes.header.frame_id = g_frame_id;
    if (save_result)
    {
        Boxes2Txt(boxes, out_labels, output_file);
    }

    updateTracks(jsk_boxes);
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "pointpillars");
    ros::NodeHandle nh;

    // 加载配置
    // const std::string DB_CONF = "../bootstrap.yaml";
    const std::string DB_CONF = "/home/ros/PointPillars_MultiHead_40FPS_ROS/bootstrap.yaml";
    YAML::Node config = YAML::LoadFile(DB_CONF);

    std::string pfe_file, backbone_file;
    if (config["UseOnnx"].as<bool>())
    {
        pfe_file = config["PfeOnnx"].as<std::string>();
        backbone_file = config["BackboneOnnx"].as<std::string>();
    }
    else
    {
        pfe_file = config["PfeTrt"].as<std::string>();
        backbone_file = config["BackboneTrt"].as<std::string>();
    }
    std::cout << backbone_file << std::endl;
    const std::string pp_config = config["ModelConfig"].as<std::string>();
    PointPillars pp(
        config["ScoreThreshold"].as<float>(),
        config["NmsOverlapThreshold"].as<float>(),
        config["UseOnnx"].as<bool>(),
        pfe_file,
        backbone_file,
        pp_config);
    std::string file_name = config["InputFile"].as<std::string>();
    std::string lidar_topic = config["LidarTopic"].as<std::string>();
    std::string bbox_topic = config["BoundingBoxTopic"].as<std::string>();
    std::string out_file_name = config["OutputFile"].as<std::string>();
    g_frame_id = config["FrameId"].as<std::string>();
    g_score_threshold = config["ObjectScoreThreshold"].as<float>();
    bool save_result = config["Save"].as<bool>();
    if (save_result)
    {
        ofstream ofFile;
        ofFile.open(out_file_name, std::ios::out);
        ofFile.close();
    }
    // 订阅lidar_topic,发布boundingbox
    ros::Subscriber lidar_sub = nh.subscribe(lidar_topic, 100, &lidar_callback);
    ros::Publisher pub_bbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>(bbox_topic, 100);
    ros::Publisher pub_tracked_bbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/tracked_boxes",100);

    std::vector<float> out_detections;
    std::vector<int> out_labels;
    std::vector<float> out_scores;

    // 推理
    ros::Rate rate(10); // 10hz
    while (ros::ok())
    {
        ros::spinOnce();
        cudaDeviceSynchronize();
        pp.DoInference(points_array, in_num_points, &out_detections, &out_labels,
                       &out_scores);
        cudaDeviceSynchronize();

        // 发布检测结果
        publishDetectionResult(out_detections, out_labels, out_scores, pub_bbox, save_result, out_file_name);

        publicTrackedBoxes(pub_tracked_bbox);

        delete points_array;
        points_array = nullptr;
        rate.sleep();
    }
    return 0;
};
