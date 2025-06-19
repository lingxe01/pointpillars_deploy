// ROS2 Humble headers
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "jsk_recognition_msgs/msg/bounding_box_array.hpp"
#include "jsk_recognition_msgs/msg/bounding_box.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// STL & Other
#include <fstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include "yaml-cpp/yaml.h"
#include "../pointpillars/pointpillars.h"

using namespace std::chrono_literals;
using namespace std;

float *points_array;
int in_num_points;
float g_score_threshold;
std::string g_frame_id;
float distance_threshold = 1.5;

struct Track {
    int id;
    jsk_recognition_msgs::msg::BoundingBox bbox;
    int age;
    int lost;
};

std::vector<Track> tracks;
int global_track_id = 0;

const std::vector<std::string> CLASS_NAMES = {
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"
};

std::string get_current_time() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

float centerDistance(const jsk_recognition_msgs::msg::BoundingBox &a,
                     const jsk_recognition_msgs::msg::BoundingBox &b) {
    float dx = a.pose.position.x - b.pose.position.x;
    float dy = a.pose.position.y - b.pose.position.y;
    float dz = a.pose.position.z - b.pose.position.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void updateTracks(const jsk_recognition_msgs::msg::BoundingBoxArray &detections) {
    std::vector<bool> matched(detections.boxes.size(), false);
    for (auto &track : tracks) {
        float min_dist = 9999;
        int best_idx = -1;
        for (size_t i = 0; i < detections.boxes.size(); i++) {
            float dist = centerDistance(track.bbox, detections.boxes[i]);
            if (dist < min_dist && dist < distance_threshold) {
                min_dist = dist;
                best_idx = i;
            }
        }
        if (best_idx != -1 && !matched[best_idx]) {
            track.bbox = detections.boxes[best_idx];
            track.age++;
            track.lost = 0;
            matched[best_idx] = true;
        } else {
            track.lost++;
        }
    }
    for (size_t i = 0; i < detections.boxes.size(); i++) {
        if (!matched[i]) {
            Track new_track;
            new_track.id = global_track_id++;
            new_track.bbox = detections.boxes[i];
            new_track.age = 1;
            new_track.lost = 0;
            tracks.push_back(new_track);
        }
    }
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
        [](const Track &track) { return track.lost > 5; }), tracks.end());
}

class PointPillarsNode : public rclcpp::Node {
public:
    PointPillarsNode() : Node("pointpillars_node") {
        YAML::Node config = YAML::LoadFile("/home/ros/PointPillars_MultiHead_40FPS_ROS/bootstrap.yaml");
        std::string pfe_file = config["PfeOnnx"].as<std::string>();
        std::string backbone_file = config["BackboneOnnx"].as<std::string>();
        std::string pp_config = config["ModelConfig"].as<std::string>();
        g_frame_id = config["FrameId"].as<std::string>();
        g_score_threshold = config["ObjectScoreThreshold"].as<float>();
        save_result = config["Save"].as<bool>();
        out_file_name = config["OutputFile"].as<std::string>();
        if (save_result) std::ofstream(out_file_name, std::ios::out).close();

        pp_ = std::make_unique<PointPillars>(
            config["ScoreThreshold"].as<float>(),
            config["NmsOverlapThreshold"].as<float>(),
            true, pfe_file, backbone_file, pp_config);

        pub_bbox_ = this->create_publisher<jsk_recognition_msgs::msg::BoundingBoxArray>(
            config["BoundingBoxTopic"].as<std::string>(), 10);
        pub_tracked_bbox_ = this->create_publisher<jsk_recognition_msgs::msg::BoundingBoxArray>(
            "/tracked_boxes", 10);

        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            config["LidarTopic"].as<std::string>(), 10,
            std::bind(&PointPillarsNode::lidarCallback, this, std::placeholders::_1));

        timer_ = this->create_wall_timer(100ms, std::bind(&PointPillarsNode::inferenceLoop, this));
    }

private:
    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *cloud);
        auto trans_xyz = cloud->getMatrixXfMap(3, 8, 0);
        auto trans_i = cloud->getMatrixXfMap(1, 8, 4);
        Eigen::MatrixXf pointsmap(trans_xyz.rows() + trans_i.rows(), trans_xyz.cols());
        pointsmap << trans_xyz, trans_i;
        Eigen::Matrix<float, Eigen::Dynamic, 4> data_in = pointsmap.transpose();
        in_num_points = data_in.rows();
        if (in_num_points < 1000) return;

        delete[] points_array;
        points_array = new float[in_num_points * 5];
        for (int i = 0; i < in_num_points; i++) {
            points_array[i*5 + 0] = data_in(i, 0);
            points_array[i*5 + 1] = data_in(i, 1);
            points_array[i*5 + 2] = data_in(i, 2);
            points_array[i*5 + 3] = data_in(i, 3);
            points_array[i*5 + 4] = 0.0f;
        }
    }

    void inferenceLoop() {
        std::vector<float> out_detections;
        std::vector<int> out_labels;
        std::vector<float> out_scores;

        if (!points_array || in_num_points < 1000) return;

        pp_->DoInference(points_array, in_num_points, &out_detections, &out_labels, &out_scores);
        publishDetectionResult(out_detections, out_labels, out_scores);
        publishTrackedBoxes();
        delete[] points_array;
        points_array = nullptr;
    }

    void publishDetectionResult(const std::vector<float> &boxes,
                                const std::vector<int> &labels,
                                const std::vector<float> &scores) {
        jsk_recognition_msgs::msg::BoundingBoxArray array;
        array.header.frame_id = g_frame_id;
        array.header.stamp = now();
        for (int i = 0; i < boxes.size(); i += 7) {
            if (scores[i / 7] < g_score_threshold) continue;
            jsk_recognition_msgs::msg::BoundingBox box;
            box.header = array.header;
            box.pose.position.x = boxes[i];
            box.pose.position.y = boxes[i + 1];
            box.pose.position.z = boxes[i + 2];
            box.dimensions.x = boxes[i + 3];
            box.dimensions.y = boxes[i + 4];
            box.dimensions.z = boxes[i + 5];
            Eigen::AngleAxisd yaw(boxes[i + 6], Eigen::Vector3d::UnitZ());
            Eigen::Quaterniond q = yaw;
            box.pose.orientation.w = q.w();
            box.pose.orientation.x = q.x();
            box.pose.orientation.y = q.y();
            box.pose.orientation.z = q.z();
            box.label = labels[i / 7];
            box.value = scores[i / 7];
            array.boxes.push_back(box);
        }
        updateTracks(array);
        pub_bbox_->publish(array);

        if (save_result) {
            std::ofstream ofFile(out_file_name, std::ios::app);
            for (size_t i = 0; i < array.boxes.size(); ++i) {
                const auto &b = array.boxes[i];
                ofFile << CLASS_NAMES[b.label] << " "
                       << b.pose.position.x << " " << b.pose.position.y << " " << b.pose.position.z << " "
                       << b.dimensions.x << " " << b.dimensions.y << " " << b.dimensions.z << " "
                       << "0.0\n"; // yaw
            }
            ofFile << "------" << get_current_time() << "------\n";
        }
    }

    void publishTrackedBoxes() {
        jsk_recognition_msgs::msg::BoundingBoxArray array;
        array.header.frame_id = g_frame_id;
        array.header.stamp = now();
        for (const auto &track : tracks) {
            auto box = track.bbox;
            box.label = track.id;
            array.boxes.push_back(box);
        }
        pub_tracked_bbox_->publish(array);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Publisher<jsk_recognition_msgs::msg::BoundingBoxArray>::SharedPtr pub_bbox_;
    rclcpp::Publisher<jsk_recognition_msgs::msg::BoundingBoxArray>::SharedPtr pub_tracked_bbox_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unique_ptr<PointPillars> pp_;
    std::string out_file_name;
    bool save_result = false;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointPillarsNode>());
    rclcpp::shutdown();
    return 0;
}
