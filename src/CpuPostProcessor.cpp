#include "CpuPostProcessor.hpp"
#include "Tag36h11.h"
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>

bool CpuPostProcessor::fitQuad(const std::vector<cv::Point2f>& borderPoints,
                                std::array<cv::Point2f, 4>& corners) {
    if (borderPoints.size() < 16) return false;

    std::vector<cv::Point> hullPts(borderPoints.begin(), borderPoints.end());
    std::vector<cv::Point> hull;
    cv::convexHull(hullPts, hull);
    if (hull.size() < 4) return false;

    std::vector<cv::Point2f> approx;
    cv::approxPolyDP(cv::Mat(hull), approx, cv::arcLength(cv::Mat(hull), true) * 0.02, true);
    if (approx.size() < 4) return false;

    cv::Point2f centroid{0.f, 0.f};
    for (auto& p : approx) { centroid.x += p.x; centroid.y += p.y; }
    centroid.x /= static_cast<float>(approx.size());
    centroid.y /= static_cast<float>(approx.size());

    std::vector<std::pair<double,int>> angIdx;
    for (int i = 0; i < static_cast<int>(approx.size()); ++i) {
        double ang = std::atan2(approx[i].y - centroid.y, approx[i].x - centroid.x);
        angIdx.push_back({ang, i});
    }
    std::sort(angIdx.begin(), angIdx.end());

    int step = static_cast<int>(std::max(1, static_cast<int>(approx.size()) / 4));
    for (int i = 0; i < 4; ++i) {
        corners[i] = approx[angIdx[(i * step) % static_cast<int>(angIdx.size())].second];
    }

    int topRight = 0;
    for (int i = 1; i < 4; ++i)
        if (corners[i].x - corners[i].y > corners[topRight].x - corners[topRight].y)
            topRight = i;

    std::array<cv::Point2f, 4> reordered{{corners[0], corners[1], corners[2], corners[3]}};
    for (int i = 0; i < 4; ++i) reordered[(topRight + i) % 4] = corners[i];

    double cp = (reordered[1].x - reordered[0].x) * (reordered[2].y - reordered[0].y)
              - (reordered[1].y - reordered[0].y) * (reordered[2].x - reordered[0].x);
    if (cp < 0) { std::swap(reordered[1], reordered[3]); }

    corners = reordered;
    return true;
}

std::vector<AprilTags::TagDetection> CpuPostProcessor::detect(
        const cv::Mat& gray, int /*cameraDecimate*/) {
    static AprilTags::TagDetector detector{AprilTags::tagCodes36h11};

    cv::Mat src = gray;
    if (gray.type() != CV_8UC1) {
        cv::cvtColor(gray, src, cv::COLOR_BGR2GRAY);
    }

    return detector.extractTags(src);
}

TagDetectionData CpuPostProcessor::toTagDetectionData(const AprilTags::TagDetection& det) {
    TagDetectionData out;
    out.id = det.id;
    out.hammingDist = det.hammingDistance;
    out.center = cv::Point2f(det.cxy.first, det.cxy.second);
    out.corners.resize(4);
    for (int i = 0; i < 4; ++i) {
        out.corners[i] = cv::Point2f(det.p[i].first, det.p[i].second);
    }
    return out;
}

bool CpuPostProcessor::estimatePose(const AprilTags::TagDetection& det, double tagSizeM,
                                    const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                                    TagDetectionData& out) {
    float halfTag = static_cast<float>(tagSizeM / 2.0);

    std::vector<cv::Point3f> objPts({
        {-halfTag, -halfTag, 0.f},
        { halfTag, -halfTag, 0.f},
        { halfTag,  halfTag, 0.f},
        {-halfTag,  halfTag, 0.f}
    });

    static const float cornerCoords[4][2] = {{-1.f, -1.f}, {1.f, -1.f}, {1.f, 1.f}, {-1.f, 1.f}};
    std::vector<cv::Point2f> imgPts(4);
    for (int i = 0; i < 4; ++i) {
        auto p = det.interpolate(cornerCoords[i][0], cornerCoords[i][1]);
        imgPts[i] = cv::Point2f(p.first, p.second);
    }

    cv::Mat rvec, tvec;
    bool ok = cv::solvePnP(objPts, imgPts, cameraMatrix, distCoeffs, rvec, tvec,
                           false, cv::SOLVEPNP_IPPE_SQUARE);
    if (!ok) return false;

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    out = toTagDetectionData(det);
    out.hasPose = true;
    out.rotationMatrix = R.clone();
    out.translation = cv::Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    return true;
}
