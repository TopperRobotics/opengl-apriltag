#ifndef DETECTION_RESULT_HPP
#define DETECTION_RESULT_HPP

#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

struct QuadCandidate {
    int label;
    std::vector<cv::Point2f> borderPoints;
    cv::Rect bbox;
    float area{0.f};
    float aspectRatio{0.f};
    cv::Point2f centroid{};
};

struct TagDetectionData {
    int id{-1};
    int hammingDist{-1};
    std::vector<cv::Point2f> corners; // pixel coords, CCW from marker's corner 0
    cv::Point2f center{};
    
    // Pose (filled by pose estimator)
    bool hasPose{false};
    cv::Mat rotationMatrix; // 3x3
    cv::Vec3d translation{}; // X Y Z in camera frame

    // Homography from tag space to pixel space
    cv::Mat homography; // 3x3
};

struct DetectionFrame {
    double timestamp{0.0};
    std::vector<TagDetectionData> detections;
};

class SharedResults {
public:
    void update(DetectionFrame frame) {
        std::lock_guard lock(mutex_);
        latest_ = std::move(frame);
        hasUpdate_ = true;
        cv_.notify_one();
    }

    DetectionFrame snapshot() {
        std::lock_guard lock(mutex_);
        bool was = hasUpdate_;
        hasUpdate_ = false;
        return was ? latest_ : DetectionFrame{};
    }

    DetectionFrame getLatest() {
        std::lock_guard lock(mutex_);
        return latest_;
    }

private:
    DetectionFrame latest_;
    bool hasUpdate_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
};

#endif // DETECTION_RESULT_HPP
