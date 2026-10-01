#include <iostream>
#include <csignal>
#include <atomic>
#include <chrono>
#include <thread>
#include <string>
#include <sstream>
#include <algorithm>
#include <filesystem>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include "mjpeg_streamer.hpp"
#include "CameraCalibration.hpp"
#include "ConfigManager.hpp"
#include "GpuDetector.hpp"
#include "CpuPostProcessor.hpp"
#include "HttpServer.hpp"
#include "DetectionResult.hpp"
#include "NetworkTables.hpp"

namespace {

std::atomic<bool> g_running{true};

void handleSignal(int) {
    g_running = false;
}

double timestampNow() {
    using clock = std::chrono::system_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

cv::Mat buildCameraMatrix(const ConfigManager& cfg, std::string calibrationPath) {
    if(std::filesystem::exists(calibrationPath + "/" + cfg.getString("camera_calibration", "camera_calibration.xml"))){
        std::cout << "Using calibrated camera matrix" << std::endl;
        cv::FileStorage fs(calibrationPath + "/" + cfg.getString("camera_calibration", "camera_calibration.xml"), cv::FileStorage::READ);
        cv::Mat cameraMatrix;
        fs["camera_matrix"] >> cameraMatrix;
        fs.release();
        return cameraMatrix;
    } else {
        std::cout << "Using default camera matrix" << std::endl;
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0, 0) = cfg.getDouble("camera_fx", 800.0);
        K.at<double>(1, 1) = cfg.getDouble("camera_fy", 800.0);
        K.at<double>(0, 2) = cfg.getDouble("camera_cx", 640.0);
        K.at<double>(1, 2) = cfg.getDouble("camera_cy", 360.0);
        return K;
    }
}

cv::Mat buildDistCoeffs(const ConfigManager& cfg, std::string calibrationPath) {
    std::cout << calibrationPath + "/" + cfg.getString("camera_calibration", "camera_calibration.xml") << std::endl;
    if(std::filesystem::exists(calibrationPath + "/" + cfg.getString("camera_calibration", "camera_calibration.xml"))){
        std::cout << "Using calibrated distortion coefficients" << std::endl;
        cv::FileStorage fs(calibrationPath + "/" + cfg.getString("camera_calibration", "camera_calibration.xml"), cv::FileStorage::READ);
        cv::Mat distCoeffs;
        fs["distortion_coefficients"] >> distCoeffs;
        fs.release();
        return distCoeffs;
    } else {
        std::cout << "Using default distortion coefficients" << std::endl;
        std::string raw = cfg.getString("dist_coeffs", "[0,0,0,0,0]");
        raw.erase(std::remove(raw.begin(), raw.end(), '['), raw.end());
        raw.erase(std::remove(raw.begin(), raw.end(), ']'), raw.end());
        std::stringstream ss(raw);
        cv::Mat dist(1, 5, CV_64F);
        for (int i = 0; i < 5; ++i) {
            std::string token;
            if (!std::getline(ss, token, ',')) token = "0";
            dist.at<double>(0, i) = std::stod(token);
        }
        return dist;
    }
}

struct Options {
    int cameraIndex = 0;
    std::string dbPath = "config.db";
    std::string shaderDir = "shaders";
    int httpPort = 8080;
    int cameraStreamPort = 8081;
    std::string webuiDir = "webui";
    std::string cameraSnapshotDir = "snapshot";
    std::string calibrationDir = "calibration";
};

Options parseArgs(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--camera" && i + 1 < argc) {
            opts.cameraIndex = std::stoi(argv[++i]);
        } else if (arg == "--db" && i + 1 < argc) {
            opts.dbPath = argv[++i];
        } else if (arg == "--shaders" && i + 1 < argc) {
            opts.shaderDir = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            opts.httpPort = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: apriltag [--camera N] [--db path] [--shaders dir] [--port N] [--webui dir] [--snapshot dir] [--calibration dir]\n";
            std::exit(0);
        } else if (arg == "--webui" && i + 1 < argc) {
            opts.webuiDir = argv[++i];
        } else if (arg == "--camera-stream-port" && i + 1 < argc) {
            opts.cameraStreamPort = std::stoi(argv[++i]);
        } else if (arg == "--snapshot" && i + 1 < argc) {
            opts.cameraSnapshotDir = argv[++i];
        } else if (arg == "--calibration" && i + 1 < argc) {
            opts.calibrationDir = argv[++i];
        }
    }
    return opts;
}

bool initGLFW() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return false;
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    return true;
}

} // anonymous namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    std::cout << cv::getBuildInformation() << "\n";

    Options opts = parseArgs(argc, argv);

    ConfigManager config;
    if (!config.open(opts.dbPath)) {
        std::cerr << "Failed to open config database: " << opts.dbPath << "\n";
        return 1;
    }

    if (!initGLFW()) return 1;

    GLFWwindow* window = glfwCreateWindow(640, 480, "apriltag-offscreen", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    while (glGetError() != GL_NO_ERROR) {}

    GpuDetector::Config gpuCfg;
    gpuCfg.width = 640;
    gpuCfg.height = 480;
    gpuCfg.windowSize = config.getInt("adaptive_threshold_win", 31);
    gpuCfg.thresholdConst = -static_cast<float>(config.getDouble("adaptive_threshold_const", 7.0));
    gpuCfg.minTagArea = config.getInt("min_tag_area", 100);

    GpuDetector gpuDetector(gpuCfg);
    try {
        gpuDetector.compileShaders(opts.shaderDir);
    } catch (const std::exception& e) {
        std::cerr << "Shader error: " << e.what() << "\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    CpuPostProcessor postProcessor;
    CpuPostProcessor::Config ppCfg;
    ppCfg.tagSizeM = config.getDouble("tag_size_m", 0.16);
    ppCfg.minTagArea = config.getInt("min_tag_area", 100);
    ppCfg.maxTagArea = config.getInt("max_tag_area", 10000);
    ppCfg.decimateFactor = config.getInt("decimate_factor", 2);
    postProcessor.setConfig(ppCfg);

    cv::Mat cameraMatrix = buildCameraMatrix(config, opts.calibrationDir);
    cv::Mat distCoeffs = buildDistCoeffs(config, opts.calibrationDir);
    const double tagSizeM = ppCfg.tagSizeM;
    const int decimateFactor = std::max(1, ppCfg.decimateFactor);

    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};

    CameraCalibration calibration;
    calibration.setCalibrationImagePath(opts.cameraSnapshotDir);
    calibration.setCalibrationOutputPath(opts.calibrationDir);

    NetworkTablesClient ntClient;
    ntClient.start(ntClient.configFromManager(config));

    SharedResults sharedResults;
    HttpServer httpServer(opts.httpPort, sharedResults, config, calibration, ntClient);
    httpServer.setWebuiDir(opts.webuiDir);
    httpServer.setCameraStreamPort(opts.cameraStreamPort);
    httpServer.start();
    std::cout << "HTTP server listening on port " << opts.httpPort << "\n";

    cv::VideoCapture capture(opts.cameraIndex);

    if (!capture.isOpened()) {
        std::cout << "Failed to open camera index " << opts.cameraIndex << "\n";

        // Generate a proper placeholder frame (640x480, 3 channels, black)
        cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::putText(frame, "Camera not found.", cv::Point(50, 240),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, "Connect camera and reboot.", cv::Point(50, 300),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        while (g_running) {
            if (httpServer.isStreaming()) {
                std::vector<uchar> buff_bgr;
                cv::imencode(".jpg", frame, buff_bgr, params);
                httpServer.streamFrame("/", std::string(buff_bgr.begin(), buff_bgr.end()));
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            glfwPollEvents();
        }
    } else {
        capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        capture.set(cv::CAP_PROP_EXPOSURE, config.getDouble("camera_exposure", -6.0));
        capture.set(cv::CAP_PROP_BRIGHTNESS, config.getDouble("camera_brightness", 0.5)); // might be 0.0-1.0 or 0-255 depending on camera
        capture.set(cv::CAP_PROP_AUTO_EXPOSURE , config.getDouble("camera_autoexposure", 0.75)); // i have no idea what this could be
        // todo: add camera autodetection

        cv::Mat frame, gray, decimated;
        while (g_running) {
            if (!capture.read(frame) || frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            cv::Mat detectGray = gray;
            if (decimateFactor > 1) {
                // Resize by an exact 1/decimateFactor instead of chaining
                // pyrDown() calls. pyrDown() always halves each dimension
                // per call, so calling it (decimateFactor - 1) times
                // actually shrinks the image by 2^(decimateFactor-1), which
                // only happens to equal `decimateFactor` when it's 1 or 2.
                // For any other configured value (e.g. 3 or 4) the working
                // image was shrunk by far more than intended, while corner
                // and center coordinates below are scaled back up by a
                // plain `* decimateFactor` -- silently producing badly
                // wrong pixel coordinates (and therefore wrong poses) for
                // every detection whenever decimate_factor wasn't 1 or 2.
                cv::resize(gray, decimated, cv::Size(), 1.0 / decimateFactor, 1.0 / decimateFactor,
                           cv::INTER_AREA);
                detectGray = decimated;
            }

            gpuDetector.dispatch(detectGray);

            // Turn the GPU's connected-component bounding boxes into
            // candidate regions of interest so the (much more expensive)
            // CPU AprilTags decoder only has to search near likely tags
            // instead of the whole frame.
            std::vector<cv::Rect> candidateRegions;
            for (const auto& comp : gpuDetector.getComponentInfo()) {
                candidateRegions.emplace_back(
                    comp.bboxMinX, comp.bboxMinY,
                    comp.bboxMaxX - comp.bboxMinX + 1,
                    comp.bboxMaxY - comp.bboxMinY + 1);
            }

            auto tagDets = postProcessor.detect(detectGray, candidateRegions, decimateFactor);

            DetectionFrame result;
            result.timestamp = timestampNow();
            result.detections.reserve(tagDets.size());

            // Note: tagDets are already rescaled to full-resolution pixel
            // coordinates by postProcessor.detect() (see cameraDecimate
            // above), so no further corner/center scaling is needed here --
            // and estimatePose()'s solvePnP call below is now operating in
            // the same coordinate space as cameraMatrix/distCoeffs, which
            // were built from a full-resolution calibration.
            for (const auto& det : tagDets) {
                if (!det.good) continue;
                TagDetectionData out;
                if (CpuPostProcessor::estimatePose(det, tagSizeM, cameraMatrix, distCoeffs, out)) {
                    result.detections.push_back(std::move(out));
                } else {
                    result.detections.push_back(CpuPostProcessor::toTagDetectionData(det));
                }
            }
            //std::cout <<" Frame timestamp: " << result.timestamp << ", detections: " << result.detections.size() << "\n";
            // Publish the frame we just built directly instead of reading it
            // back out of sharedResults; that avoided an extra mutex lock and
            // a full deep copy of the frame (including cv::Mat pose data) on
            // every single detection cycle, which added needless latency to
            // the data sent to the roboRIO.
            ntClient.publishDetections(result);
            sharedResults.update(std::move(result));

            if(httpServer.isStreaming()) [[likely]] {
                std::vector<uchar> buff_bgr;
                cv::imencode(".jpg", frame, buff_bgr, params);
                httpServer.streamFrame("/", std::string(buff_bgr.begin(), buff_bgr.end()));
            }

            if(httpServer.isCameraSettingsRefreshQueued()) [[unlikely]] {
                // get new settings from config and apply to camera
                std::cout << "Refreshing camera settings from config...\n";
                std::cout << "New exposure: " << config.getDouble("camera_exposure", -6.0) << "\n";
                std::cout << "New brightness: " << config.getDouble("camera_brightness", 0.5) << "\n";
                std::cout << "New autoexposure value: " << config.getDouble("camera_autoexposure", 0.75) << "\n";
                capture.set(cv::CAP_PROP_EXPOSURE, config.getDouble("camera_exposure", -6.0));
                capture.set(cv::CAP_PROP_BRIGHTNESS, config.getDouble("camera_brightness", 0.5));
                capture.set(cv::CAP_PROP_AUTO_EXPOSURE , config.getDouble("camera_autoexposure", 0.75)); // i have no idea what this could be
                httpServer.clearCameraSettingsRefreshQueue();
            }

            if(httpServer.isCameraSnapshotQueued()) [[unlikely]] {
                std::cout << "Saving snapshot to " << opts.cameraSnapshotDir + "/image" + std::to_string(result.timestamp) + ".jpg" << std::endl;
                cv::imwrite(opts.cameraSnapshotDir + "/image" + std::to_string(result.timestamp) + ".jpg", frame);
                httpServer.clearCameraSnapshotQueue();
            }

            glfwPollEvents();
        }
    }

    httpServer.stop();
    capture.release();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
