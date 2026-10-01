#ifndef CPU_POSTPROCESSOR_HPP
#define CPU_POSTPROCESSOR_HPP

#include <array>
#include <vector>
#include <opencv2/opencv.hpp>
#include "TagDetector.h"
#include "TagFamily.h"
#include "DetectionResult.hpp"

class CpuPostProcessor {
public:
    struct Config {
        double tagSizeM{0.16};
        int minTagArea{100};
        int maxTagArea{10000};
        int decimateFactor{2};
    };

    void setConfig(const Config& cfg) { config_ = cfg; }

    // Detects AprilTags in `gray`. If `candidateRegions` is non-empty, it is
    // treated as a set of GPU-generated regions of interest (e.g. connected
    // components from an adaptive-threshold + CCL pass) that are likely to
    // contain a tag; the (trusted, full-accuracy) AprilTags decoder is run
    // only on padded crops around those regions instead of the whole frame,
    // which is significantly cheaper on the CPU. If `candidateRegions` is
    // empty (or the regions collectively cover most of the frame / there
    // are too many of them to be worth splitting up), the whole frame is
    // scanned instead, guaranteeing detection never regresses when the GPU
    // pipeline hasn't run yet (e.g. first frame) or produces no candidates.
    //
    // `cameraDecimate` must be set to the factor by which `gray` has been
    // downscaled relative to the camera's native resolution (1 if it
    // hasn't been decimated at all). All pixel-space fields of the
    // returned detections (corners, center, homography) are rescaled back
    // up to full-resolution coordinates before being returned, so they can
    // be used directly with a camera matrix calibrated at full resolution
    // (e.g. by estimatePose()).
    std::vector<AprilTags::TagDetection> detect(
        const cv::Mat& gray,
        const std::vector<cv::Rect>& candidateRegions = {},
        int cameraDecimate = 1);

    static bool fitQuad(const std::vector<cv::Point2f>& borderPoints,
                        std::array<cv::Point2f, 4>& corners);

    static TagDetectionData toTagDetectionData(const AprilTags::TagDetection& det);

    static bool estimatePose(const AprilTags::TagDetection& det,
                             double tagSizeM,
                             const cv::Mat& cameraMatrix,
                             const cv::Mat& distCoeffs,
                             TagDetectionData& out);

private:
    Config config_;
};

#endif // CPU_POSTPROCESSOR_HPP
