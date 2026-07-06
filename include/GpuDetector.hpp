#ifndef GPU_DETECTOR_HPP
#define GPU_DETECTOR_HPP

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class GpuDetector {
public:
    struct Config {
        int width{640};
        int height{480};
        int windowSize{31};
        float thresholdConst{-7.0f};
        int numComponentsMax{4096};
        int minTagArea{100};
    };

    GpuDetector(const Config& cfg);
    ~GpuDetector();

    GpuDetector(const GpuDetector&) = delete;
    GpuDetector& operator=(const GpuDetector&) = delete;

    bool compileShaders(const std::string& shaderDir);
    
    int dispatch(const cv::Mat& grayImage);

    struct ComponentInfo {
        uint32_t count;
        uint32_t offset;
        int32_t  bboxMinX, bboxMinY;
        int32_t  bboxMaxX, bboxMaxY;
        float    centroidX, centroidY;
    };

    std::vector<ComponentInfo> getComponentInfo() const;
    std::vector<uint32_t> getBorderPoints() const;
    bool resize(int width, int height);

private:
    Config config_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // GPU_DETECTOR_HPP
