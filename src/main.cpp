#include <iostream>
#include <csignal>
#include <atomic>
#include <chrono>
#include <thread>
#include <string>
#include <sstream>
#include <algorithm>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include "mjpeg_streamer.hpp"

#include "ConfigManager.hpp"
#include "GpuDetector.hpp"
#include "CpuPostProcessor.hpp"
#include "HttpServer.hpp"
#include "DetectionResult.hpp"

namespace {

std::atomic<bool> g_running{true};

void handleSignal(int) {
    g_running = false;
}

double timestampNow() {
    using clock = std::chrono::system_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

cv::Mat buildCameraMatrix(const ConfigManager& cfg) {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = cfg.getDouble("camera_fx", 800.0);
    K.at<double>(1, 1) = cfg.getDouble("camera_fy", 800.0);
    K.at<double>(0, 2) = cfg.getDouble("camera_cx", 640.0);
    K.at<double>(1, 2) = cfg.getDouble("camera_cy", 360.0);
    return K;
}

cv::Mat buildDistCoeffs(const ConfigManager& cfg) {
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

struct Options {
    int cameraIndex = 0;
    std::string dbPath = "config.db";
    std::string shaderDir = "shaders";
    int httpPort = 8080;
    int cameraStreamPort = 8081;
    std::string webuiDir = "webui";
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
            std::cout << "Usage: apriltag [--camera N] [--db path] [--shaders dir] [--port N] [--webui dir]\n";
            std::exit(0);
        } else if (arg == "--webui" && i + 1 < argc) {
            opts.webuiDir = argv[++i];
        } else if (arg == "--camera-stream-port" && i + 1 < argc) {
            opts.cameraStreamPort = std::stoi(argv[++i]);
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

    cv::Mat cameraMatrix = buildCameraMatrix(config);
    cv::Mat distCoeffs = buildDistCoeffs(config);
    const double tagSizeM = ppCfg.tagSizeM;
    const int decimateFactor = std::max(1, ppCfg.decimateFactor);

    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90}; // todo: make configurable, add an option to sqlite db

    SharedResults sharedResults;
    HttpServer httpServer(opts.httpPort, sharedResults, config);
    httpServer.setWebuiDir(opts.webuiDir);
    httpServer.setCameraStreamPort(opts.cameraStreamPort);
    httpServer.start();
    std::cout << "HTTP server listening on port " << opts.httpPort << "\n";

    cv::VideoCapture capture(opts.cameraIndex);

    capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    capture.set(cv::CAP_PROP_EXPOSURE, config.getDouble("camera_exposure", -6.0));
    capture.set(cv::CAP_PROP_BRIGHTNESS, config.getDouble("camera_brightness", 0.5)); // might be 0.0-1.0 or 0-255 depending on camera
    capture.set(cv::CAP_PROP_AUTO_EXPOSURE , config.getDouble("camera_autoexposure", 0.75)); // i have no idea what this could be
    if (!capture.isOpened()) {
        std::cerr << "Failed to open camera index " << opts.cameraIndex << "\n";
        httpServer.stop();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    cv::Mat frame, gray, decimated;
    while (g_running) {
        if (!capture.read(frame) || frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        cv::Mat detectGray = gray;
        if (decimateFactor > 1) {
            cv::Mat tmp = gray;
            for (int i = 1; i < decimateFactor; ++i) {
                cv::pyrDown(tmp, decimated);
                tmp = decimated;
            }
            detectGray = decimated;
        }

        gpuDetector.dispatch(detectGray);

        auto tagDets = postProcessor.detect(detectGray, decimateFactor);

        DetectionFrame result;
        result.timestamp = timestampNow();
        result.detections.reserve(tagDets.size());

        for (const auto& det : tagDets) {
            if (!det.good) continue;
            TagDetectionData out;
            if (CpuPostProcessor::estimatePose(det, tagSizeM, cameraMatrix, distCoeffs, out)) {
                if (decimateFactor > 1) {
                    float scale = static_cast<float>(decimateFactor);
                    for (auto& c : out.corners) {
                        c.x *= scale;
                        c.y *= scale;
                    }
                    out.center.x *= scale;
                    out.center.y *= scale;
                }
                result.detections.push_back(std::move(out));
            } else {
                TagDetectionData basic = CpuPostProcessor::toTagDetectionData(det);
                if (decimateFactor > 1) {
                    float scale = static_cast<float>(decimateFactor);
                    for (auto& c : basic.corners) {
                        c.x *= scale;
                        c.y *= scale;
                    }
                    basic.center.x *= scale;
                    basic.center.y *= scale;
                }
                result.detections.push_back(std::move(basic));
            }
        }
        //std::cout <<" Frame timestamp: " << result.timestamp << ", detections: " << result.detections.size() << "\n";
        sharedResults.update(std::move(result));

        if(httpServer.isStreaming()) {
            std::vector<uchar> buff_bgr;
            cv::imencode(".jpg", frame, buff_bgr, params);
            httpServer.streamFrame("/", std::string(buff_bgr.begin(), buff_bgr.end()));
        }

        if(httpServer.isCameraSettingsRefreshQueued()) {
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

        glfwPollEvents();
    }

    httpServer.stop();
    capture.release();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
