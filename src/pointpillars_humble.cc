// headers in STL
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <memory>
#include <vector>
#include <algorithm>

// ros2
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "jsk_recognition_msgs/msg/bounding_box_array.hpp"
#include "jsk_recognition_msgs/msg/bounding_box.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

// pcl
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// yaml
#include <yaml-cpp/yaml.h>

// headers in 3rd-part
#include "../pointpillars/pointpillars.h"

using namespace std;
using namespace std::chrono_literals;

struct Track
{
    int id;
    jsk_recognition_msgs::msg::BoundingBox bbox;
    int age;
    int lost;
};

class PointPillarsNode : public rclcpp::Node
{
public:
    PointPillarsNode() : Node("pointpillars")
    {
        // 声明并获取参数
        this->declare_parameter("config_file", "/home/ros/PointPillars_MultiHead_40FPS_ROS/bootstrap.yaml");
        std::string config_file = this->get_parameter("config_file").as_string();
        
        // 加载配置
        try {
            config_ = YAML::LoadFile(config_file);
        } catch (const YAML::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load config file: %s", e.what());
            return;
        }
        
        // 初始化参数
        std::string pfe_file, backbone_file;
        if (config_["UseOnnx"].as<bool>())
        {
            pfe_file = config_["PfeOnnx"].as<std::string>();
            backbone_file = config_["BackboneOnnx"].as<std::string>();
        }
        else
        {
            pfe_file = config_["PfeTrt"].as<std::string>();
            backbone_file = config_["BackboneTrt"].as<std::string>();
        }
        
        RCLCPP_INFO(this->get_logger(), "Backbone file: %s", backbone_file.c_str());
        
        const std::string pp_config = config_["ModelConfig"].as<std::string>();
        pp_ = std::make_unique<PointPillars>(
            config_["ScoreThreshold"].as<float>(),
            config_["NmsOverlapThreshold"].as<float>(),
            config_["UseOnnx"].as<bool>(),
            pfe_file,
            backbone_file,
            pp_config);
        
        file_name_ = config_["InputFile"].as<std::string>();
        lidar_topic_ = config_["LidarTopic"].as<std::string>();
        bbox_topic_ = config_["BoundingBoxTopic"].as<std::string>();
        out_file_name_ = config_["OutputFile"].as<std::string>();
        g_frame_id_ = config_["FrameId"].as<std::string>();
        g_score_threshold_ = config_["ObjectScoreThreshold"].as<float>();
        save_result_ = config_["Save"].as<bool>();
        
        if (save_result_)
        {
            ofstream ofFile;
            ofFile.open(out_file_name_, std::ios::out);
            ofFile.close();
        }
        
        // 创建订阅者和发布者
        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            lidar_topic_, 100, std::bind(&PointPillarsNode::lidar_callback, this, std::placeholders::_1));
        
        pub_bbox_ = this->create_publisher<jsk_recognition_msgs::msg::BoundingBoxArray>(bbox_topic_, 100);
        pub_tracked_bbox_ = this->create_publisher<jsk_recognition_msgs::msg::BoundingBoxArray>("/tracked_boxes", 100);
        
        // 创建定时器
        timer_ = this->create_wall_timer(
            100ms, std::bind(&PointPillarsNode::timer_callback, this));
        
        RCLCPP_INFO(this->get_logger(), "PointPillars node initialized");
    }
    
private:
    // 常量和配置
    const std::vector<std::string> CLASS_NAMES = {
        "car", "truck", "construction_vehicle", "bus", "trailer",
        "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"};
    
    // 节点参数
    YAML::Node config_;
    std::string file_name_;
    std::string lidar_topic_;
    std::string bbox_topic_;
    std::string out_file_name_;
    std::string g_frame_id_;
    float g_score_threshold_;
    bool save_result_;
    
    // 距离阈值
    float distance_threshold = 1.5;
    
    // 跟踪相关
    std::vector<Track> tracks_;
    int global_track_id = 0;
    
    // 数据
    float *points_array = nullptr;
    int in_num_points = 0;
    
    // 3D检测模型
    std::unique_ptr<PointPillars> pp_;
    
    // ROS2组件
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Publisher<jsk_recognition_msgs::msg::BoundingBoxArray>::SharedPtr pub_bbox_;
    rclcpp::Publisher<jsk_recognition_msgs::msg::BoundingBoxArray>::SharedPtr pub_tracked_bbox_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    // 工具函数
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
    
    void Txt2Arrary(float *&points_array, string file_name, int num_feature = 4)
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
    }
    
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
    }
    
    float centerDistance(const jsk_recognition_msgs::msg::BoundingBox &a,
                         const jsk_recognition_msgs::msg::BoundingBox &b)
    {
        float dx = a.pose.position.x - b.pose.position.x;
        float dy = a.pose.position.y - b.pose.position.y;
        float dz = a.pose.position.z - b.pose.position.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    
    void updateTracks(const jsk_recognition_msgs::msg::BoundingBoxArray &detections)
    {
        std::vector<bool> matched(detections.boxes.size(), false);
        for (auto &track : tracks_)
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
                tracks_.push_back(new_track);
            }
        }
        // 移除lost太久的轨迹
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                    [](const Track &track)
                                    { return track.lost > 5; }),
                     tracks_.end());
    }
    
    void publicTrackedBoxes()
    {
        jsk_recognition_msgs::msg::BoundingBoxArray tracked_array;
        tracked_array.header.frame_id = g_frame_id_;
        tracked_array.header.stamp = this->get_clock()->now();
        for(const auto& t : tracks_)
        {
            jsk_recognition_msgs::msg::BoundingBox box = t.bbox;
            box.label = t.id;
            tracked_array.boxes.push_back(box);
        }
        pub_tracked_bbox_->publish(tracked_array);
    }
    
    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pc_msg_ptr)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*pc_msg_ptr, *cloud);

        auto trans_cloudxyz = cloud->getMatrixXfMap(3, 8, 0);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> trans_cloudi = cloud->getMatrixXfMap(1, 8, 4);
        Eigen::MatrixXf pointsmap(trans_cloudxyz.rows() + trans_cloudi.rows(), trans_cloudxyz.cols());
        pointsmap << trans_cloudxyz, trans_cloudi;
        Eigen::Matrix<float, Eigen::Dynamic, 4> data_in = pointsmap.transpose();
        RCLCPP_INFO(this->get_logger(), "Lidar point size: %d", data_in.rows());
        in_num_points = data_in.rows();
        if (in_num_points < 1000)
            return;

        if (points_array != nullptr) {
            delete[] points_array;
        }
        
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
                                const std::vector<float> &out_scores)
    {
        int box_size = boxes.size() / 7;
        RCLCPP_INFO(this->get_logger(), "Inference boxes size: %d", box_size);
        if (box_size == 0)
            return;
        assert(out_labels.size() == out_scores.size());
        assert(box_size == out_labels.size());

        jsk_recognition_msgs::msg::BoundingBoxArray jsk_boxes;
        // box_dim： x，y，z，dx，dy，dz，yaw
        for (int i = 0; i < boxes.size(); i = i + 7)
        {
            jsk_recognition_msgs::msg::BoundingBox jsk_box;
            jsk_box.header.frame_id = g_frame_id_;
            jsk_box.pose.position.x = boxes[i + 0];
            jsk_box.pose.position.y = boxes[i + 1];
            jsk_box.pose.position.z = boxes[i + 2];
            jsk_box.dimensions.x = boxes[i + 3];
            jsk_box.dimensions.y = boxes[i + 4];
            jsk_box.dimensions.z = boxes[i + 5];
            // yaw
            tf2::Quaternion quaternion;
            quaternion.setRPY(0, 0, boxes[i + 6]);
            jsk_box.pose.orientation = tf2::toMsg(quaternion);
            
            int box_idx = i / 7;
            jsk_box.label = out_labels[box_idx];
            jsk_box.value = out_scores[box_idx];
            // 利用分数过滤
            if (jsk_box.value > g_score_threshold_)
            {
                jsk_boxes.boxes.emplace_back(jsk_box);
            }
        }
        jsk_boxes.header.frame_id = g_frame_id_;
        jsk_boxes.header.stamp = this->get_clock()->now();
        
        if (save_result_)
        {
            Boxes2Txt(boxes, out_labels, out_file_name_);
        }

        updateTracks(jsk_boxes);
        pub_bbox_->publish(jsk_boxes);
    }
    
    void timer_callback()
    {
        if (points_array == nullptr || in_num_points == 0) {
            return;
        }
        
        std::vector<float> out_detections;
        std::vector<int> out_labels;
        std::vector<float> out_scores;

        // 推理
        pp_->DoInference(points_array, in_num_points, &out_detections, &out_labels, &out_scores);

        // 发布检测结果
        publishDetectionResult(out_detections, out_labels, out_scores);

        // 发布跟踪结果
        publicTrackedBoxes();

        // 释放内存
        delete[] points_array;
        points_array = nullptr;
        in_num_points = 0;
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointPillarsNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}