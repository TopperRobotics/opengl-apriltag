#include "CpuPostProcessor.hpp"
#include "Tag36h11.h"
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>

namespace {

// Merge any rectangles that overlap into their bounding union. This keeps
// us from scanning near-duplicate/overlapping ROIs separately, and ensures
// a tag whose black border was split into two adjacent GPU components (or
// whose padding causes neighboring components to overlap) still ends up
// fully inside a single crop.
std::vector<cv::Rect> mergeOverlappingRects(std::vector<cv::Rect> rects) {
    bool merged = true;
    while (merged) {
        merged = false;
        for (size_t i = 0; i < rects.size() && !merged; ++i) {
            for (size_t j = i + 1; j < rects.size(); ++j) {
                if ((rects[i] & rects[j]).area() > 0) {
                    rects[i] = rects[i] | rects[j];
                    rects.erase(rects.begin() + static_cast<long>(j));
                    merged = true;
                    break;
                }
            }
        }
    }
    return rects;
}

// Translate all pixel-space fields of a detection that was produced from a
// cropped sub-image so that they refer to coordinates in the original,
// full-size image. AprilTags::TagDetection::interpolate() computes pixel
// coordinates as `homography(x,y) + hxy`, so shifting `hxy` (in addition to
// the already-resolved corner/center points) by the crop's origin is
// sufficient to make every downstream consumer (interpolate(),
// getRelativeTransform(), etc.) operate in full-image coordinates.
void offsetDetection(AprilTags::TagDetection& det, float dx, float dy) {
    det.hxy.first += dx;
    det.hxy.second += dy;
    det.cxy.first += dx;
    det.cxy.second += dy;
    for (auto& pt : det.p) {
        pt.first += dx;
        pt.second += dy;
    }
}

// Scale all pixel-space fields of a detection by `scale`, converting it from
// the coordinate space of a decimated/downscaled working image back to the
// coordinate space of the original, full-resolution frame.
// AprilTags::TagDetection::interpolate() computes pixel coordinates as:
//   interpolate(x,y) = (H(0,:)Â·[x,y,1], H(1,:)Â·[x,y,1]) / z  +  hxy
//   where z = H(2,:)Â·[x,y,1]
// Scaling the numerator rows (0 and 1) of the homography by `scale`, along
// with `hxy`, therefore scales interpolate()'s output by `scale` too,
// without touching the projective denominator (row 2). This keeps
// interpolate(), p[], and cxy all mutually consistent after rescaling.
void scaleDetection(AprilTags::TagDetection& det, float scale) {
    if (scale == 1.0f) return;
    det.hxy.first *= scale;
    det.hxy.second *= scale;
    det.cxy.first *= scale;
    det.cxy.second *= scale;
    for (auto& pt : det.p) {
        pt.first *= scale;
        pt.second *= scale;
    }
    det.homography.row(0) *= static_cast<double>(scale);
    det.homography.row(1) *= static_cast<double>(scale);
}

// When candidate regions overlap, the same physical tag can be decoded more
// than once (once per crop it falls into). Collapse those duplicates using
// the same "same id + overlapping quads -> keep the lower hamming distance,
// then greater observed perimeter" rule that AprilTags::TagDetector itself
// uses to de-duplicate detections from overlapping quad searches.
void mergeDuplicateDetections(std::vector<AprilTags::TagDetection>& detections) {
    std::vector<AprilTags::TagDetection> merged;
    merged.reserve(detections.size());
    for (const auto& det : detections) {
        bool duplicate = false;
        for (auto& existing : merged) {
            if (existing.id != det.id || !existing.overlapsTooMuch(det)) continue;
            duplicate = true;
            if (det.hammingDistance < existing.hammingDistance ||
                (det.hammingDistance == existing.hammingDistance &&
                 det.observedPerimeter > existing.observedPerimeter)) {
                existing = det;
            }
            break;
        }
        if (!duplicate) merged.push_back(det);
    }
    detections = std::move(merged);
}

} // namespace

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
        const cv::Mat& gray, const std::vector<cv::Rect>& candidateRegions,
        int cameraDecimate) {
    static AprilTags::TagDetector detector{AprilTags::tagCodes36h11};

    cv::Mat src = gray;
    if (gray.type() != CV_8UC1) {
        cv::cvtColor(gray, src, cv::COLOR_BGR2GRAY);
    }

    const cv::Rect fullFrame(0, 0, src.cols, src.rows);
    std::vector<AprilTags::TagDetection> result;

    if (candidateRegions.empty()) {
        // No GPU candidates available (pipeline not running / first frame)
        // -- fall back to scanning the whole frame so detection accuracy
        // never regresses.
        result = detector.extractTags(src);
    } else {
        // The GPU connected-component bounding box only covers the
        // thresholded black pixels of the tag's border. AprilTags' quad
        // search needs to see the white quiet zone around it too, so pad
        // generously relative to the component's own size (with a fixed
        // minimum) before cropping.
        std::vector<cv::Rect> padded;
        padded.reserve(candidateRegions.size());
        for (const auto& r : candidateRegions) {
            int pad = std::max({ static_cast<int>(r.width * 0.75f),
                                  static_cast<int>(r.height * 0.75f), 24 });
            cv::Rect p(r.x - pad, r.y - pad, r.width + 2 * pad, r.height + 2 * pad);
            p &= fullFrame;
            if (p.width > 0 && p.height > 0) padded.push_back(p);
        }

        padded = mergeOverlappingRects(std::move(padded));

        // If the merged regions already cover most of the frame, or there
        // are too many of them, just run the detector over the whole image
        // once: splitting it up would cost more than it saves (and adds
        // needless complexity/risk) once the candidate set stops being
        // "sparse".
        double totalArea = 0.0;
        for (const auto& r : padded) totalArea += r.area();
        const double frameArea = static_cast<double>(fullFrame.area());
        constexpr size_t kMaxRegions = 12;
        if (padded.empty() || padded.size() > kMaxRegions ||
            (frameArea > 0.0 && totalArea > 0.6 * frameArea)) {
            result = detector.extractTags(src);
        } else {
            for (const auto& region : padded) {
                // AprilTags::TagDetector::extractTags() walks image.data
                // linearly (image.data[i], i incrementing once per pixel)
                // without ever consulting cv::Mat::step/isContinuous(). A
                // plain ROI view (src(region)) is *not* contiguous unless
                // it happens to span the full width of `src` -- its row
                // stride is still src's full row width, not the crop's --
                // so feeding it straight into extractTags() would silently
                // read misaligned/garbage pixel data for every row beyond
                // the first. clone() makes a tightly-packed, contiguous
                // copy so the crop is decoded correctly.
                cv::Mat crop = src(region).clone();
                auto dets = detector.extractTags(crop);
                for (auto& det : dets) {
                    offsetDetection(det, static_cast<float>(region.x), static_cast<float>(region.y));
                    result.push_back(det);
                }
            }
            mergeDuplicateDetections(result);
        }
    }

    // Every field above is in the coordinate space of `gray`, which the
    // caller may have decimated/downscaled by `cameraDecimate` before
    // detection (for performance). Rescale back to the original,
    // full-resolution frame here so that callers can feed these detections
    // straight into solvePnP-based pose estimation using a camera matrix
    // calibrated at full resolution. This parameter used to be accepted but
    // silently ignored, which meant estimatePose() combined decimated-image
    // pixel coordinates with a full-resolution camera matrix -- silently
    // producing wrong translation/rotation for every single detection
    // whenever decimation was enabled (the default configuration is
    // decimate_factor = 2).
    if (cameraDecimate > 1) {
        const float scale = static_cast<float>(cameraDecimate);
        for (auto& det : result) {
            scaleDetection(det, scale);
        }
    }

    return result;
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
