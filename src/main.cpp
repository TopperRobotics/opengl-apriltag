#include <iostream>
#include <csignal>
#include <atomic>
#include <chrono>
#include <thread>
#include <string>
#include <sstream>
#include <algorithm>
#include <filesystem>

// libepoxy provides the GL and EGL entry points. <epoxy/egl.h> transitively
// includes <EGL/egl.h>, <EGL/eglext.h>, and <epoxy/gl.h>, so no separate
// GL or EGL headers are needed here. Unlike GLEW there is no init call and
// no "experimental" mode to enable -- entry points are resolved lazily on
// first use.
#include <epoxy/gl.h>
#include <epoxy/egl.h>

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

// ---------------------------------------------------------------------------
// CameraManager: Encapsulates camera connection, auto-detection via driver,
// and settings application.
// ---------------------------------------------------------------------------
class CameraManager {
public:
    CameraManager() = default;
    ~CameraManager() { release(); }

    bool initialize(int preferredIndex, const ConfigManager& config) {
        int cameraIndex = preferredIndex;
        if (!tryOpen(cameraIndex)) {
            std::cout << "Preferred camera index " << cameraIndex << " failed. Starting camera autodetection...\n";
            cameraIndex = detectAvailableCamera();
            if (cameraIndex < 0 || !tryOpen(cameraIndex)) {
                std::cerr << "Camera autodetection failed: No functional camera found.\n";
                return false;
            }
        }

        applySettings(config);
        return true;
    }

    bool isOpened() const {
        return capture_.isOpened();
    }

    bool read(cv::Mat& frame) {
        return capture_.read(frame);
    }

    void applySettings(const ConfigManager& config) {
        if (!capture_.isOpened()) return;

        capture_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        capture_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        capture_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

        // V4L2 Auto Exposure mapping: 3.0 = V4L2_EXPOSURE_AUTO, 1.0 = V4L2_EXPOSURE_MANUAL
        double autoExposureVal = config.getDouble("camera_autoexposure", 3.0);
        capture_.set(cv::CAP_PROP_AUTO_EXPOSURE, autoExposureVal);

        capture_.set(cv::CAP_PROP_EXPOSURE, config.getDouble("camera_exposure", -6.0));
        capture_.set(cv::CAP_PROP_BRIGHTNESS, config.getDouble("camera_brightness", 0.5));

        std::cout << "Camera settings applied. Index: " << activeIndex_
                  << " | AutoExposure: " << autoExposureVal
                  << " | Exposure: " << config.getDouble("camera_exposure", -6.0)
                  << " | Brightness: " << config.getDouble("camera_brightness", 0.5) << "\n";
    }

    void release() {
        if (capture_.isOpened()) {
            capture_.release();
            activeIndex_ = -1;
        }
    }

private:
    bool tryOpen(int index) {
        std::cout << "Attempting to open camera at index " << index << "...\n";
        capture_.open(index, cv::CAP_ANY);
        if (capture_.isOpened()) {
            activeIndex_ = index;
            std::cout << "Successfully opened camera at index " << index << "\n";
            return true;
        }
        return false;
    }

    int detectAvailableCamera() {
        // Query Linux driver nodes /dev/video*
        for (int i = 0; i < 16; ++i) {
            std::string devPath = "/dev/video" + std::to_string(i);
            if (std::filesystem::exists(devPath)) {
                cv::VideoCapture testCap(i);
                if (testCap.isOpened()) {
                    std::cout << "Autodetected active camera driver node at " << devPath << " (index " << i << ")\n";
                    testCap.release();
                    return i;
                }
            }
        }

        // Fallback: brute-force standard indices 0 to 4
        for (int i = 0; i < 5; ++i) {
            cv::VideoCapture testCap(i);
            if (testCap.isOpened()) {
                std::cout << "Autodetected working camera at fallback index " << i << "\n";
                testCap.release();
                return i;
            }
        }

        return -1;
    }

    cv::VideoCapture capture_;
    int activeIndex_ = -1;
};

// ---------------------------------------------------------------------------
// Headless EGL-backed OpenGL context.
// ---------------------------------------------------------------------------
struct HeadlessGLContext {
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;
    EGLConfig  config  = nullptr;

    bool init(int width, int height) {
        display = EGL_NO_DISPLAY;

        auto eglQueryDevicesEXT = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(
            eglGetProcAddress("eglQueryDevicesEXT"));
        auto eglGetPlatformDisplayEXT = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
            eglGetProcAddress("eglGetPlatformDisplayEXT"));

        if (eglQueryDevicesEXT && eglGetPlatformDisplayEXT) {
            constexpr EGLint kMaxDevices = 16;
            EGLDeviceEXT devices[kMaxDevices];
            EGLint numDevices = 0;
            if (eglQueryDevicesEXT(kMaxDevices, devices, &numDevices) == EGL_TRUE && numDevices > 0) {
                for (EGLint i = 0; i < numDevices; ++i) {
                    EGLDisplay d = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
                    if (d == EGL_NO_DISPLAY) continue;
                    EGLint major = 0, minor = 0;
                    if (eglInitialize(d, &major, &minor) == EGL_TRUE) {
                        display = d;
                        std::cout << "EGL: initialized device display (EGL "
                                  << major << "." << minor << ")\n";
                        break;
                    }
                }
            }
        }

        if (display == EGL_NO_DISPLAY) {
            display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
            if (display == EGL_NO_DISPLAY) {
                std::cerr << "EGL: eglGetDisplay(EGL_DEFAULT_DISPLAY) failed\n";
                return false;
            }
            EGLint major = 0, minor = 0;
            if (eglInitialize(display, &major, &minor) != EGL_TRUE) {
                std::cerr << "EGL: eglInitialize failed: 0x"
                          << std::hex << eglGetError() << std::dec << "\n";
                display = EGL_NO_DISPLAY;
                return false;
            }
            std::cout << "EGL: initialized default display (EGL "
                      << major << "." << minor << ")\n";
        }

        if (eglBindAPI(EGL_OPENGL_API) != EGL_TRUE) {
            std::cerr << "EGL: eglBindAPI(EGL_OPENGL_API) failed: 0x"
                      << std::hex << eglGetError() << std::dec << "\n";
            return false;
        }

        EGLint configAttribs[] = {
            EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_RED_SIZE,        8,
            EGL_GREEN_SIZE,      8,
            EGL_BLUE_SIZE,       8,
            EGL_ALPHA_SIZE,      8,
            EGL_DEPTH_SIZE,      24,
            EGL_NONE
        };
        EGLint numConfigs = 0;
        if (eglChooseConfig(display, configAttribs, &config, 1, &numConfigs) != EGL_TRUE
            || numConfigs < 1) {
            std::cerr << "EGL: eglChooseConfig failed: 0x"
                      << std::hex << eglGetError() << std::dec << "\n";
            return false;
        }

        EGLint contextAttribs[] = {
            EGL_CONTEXT_MAJOR_VERSION, 4,
            EGL_CONTEXT_MINOR_VERSION, 3,
            EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            EGL_NONE
        };
        context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
        if (context == EGL_NO_CONTEXT) {
            EGLint fallbackAttribs[] = { EGL_NONE };
            context = eglCreateContext(display, config, EGL_NO_CONTEXT, fallbackAttribs);
        }
        if (context == EGL_NO_CONTEXT) {
            std::cerr << "EGL: eglCreateContext failed: 0x"
                      << std::hex << eglGetError() << std::dec << "\n";
            return false;
        }

        EGLint pbufferAttribs[] = {
            EGL_WIDTH,  width,
            EGL_HEIGHT, height,
            EGL_NONE
        };
        surface = eglCreatePbufferSurface(display, config, pbufferAttribs);
        if (surface == EGL_NO_SURFACE) {
            std::cerr << "EGL: eglCreatePbufferSurface failed: 0x"
                      << std::hex << eglGetError() << std::dec << "\n";
            return false;
        }

        if (eglMakeCurrent(display, surface, surface, context) != EGL_TRUE) {
            std::cerr << "EGL: eglMakeCurrent failed: 0x"
                      << std::hex << eglGetError() << std::dec << "\n";
            return false;
        }

        const unsigned char* version  = glGetString(GL_VERSION);
        const unsigned char* renderer = glGetString(GL_RENDERER);
        std::printf("EGL: OpenGL context ready, Version: %s, Renderer: %s \n",
                    version ? reinterpret_cast<const char*>(version) : "<null>",
                    renderer ? reinterpret_cast<const char*>(renderer) : "<null>");
        return true;
    }

    void destroy() {
        if (display == EGL_NO_DISPLAY) return;
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (surface != EGL_NO_SURFACE) eglDestroySurface(display, surface);
        if (context != EGL_NO_CONTEXT) eglDestroyContext(display, context);
        eglTerminate(display);
        surface = EGL_NO_SURFACE;
        context = EGL_NO_CONTEXT;
        display = EGL_NO_DISPLAY;
    }
};

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

    HeadlessGLContext glContext;
    if (!glContext.init(640, 480)) {
        std::cerr << "Failed to create headless EGL OpenGL context\n";
        return 1;
    }

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
        glContext.destroy();
        return 1;
    }

    CpuPostProcessor postProcessor;
    CpuPostProcessor::Config ppCfg;
    ppCfg.tagSizeM = config.getDouble("tag_size_m", 0.16);
    ppCfg.minTagArea = config.getInt("min_tag_area", 100);
    ppCfg.maxTagArea = config.getInt("max_tag_area", 10000);
    ppCfg.decimateFactor = config.getInt("decimate_factor", 3);
    // todo: expose to webui
    // implement like camera settings
    // especially the settings refresh queue
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

    CameraManager cameraManager;
    if (!cameraManager.initialize(opts.cameraIndex, config)) {
        std::cout << "Failed to initialize camera. Entering fallback placeholder loop...\n";

        cv::Mat placeholderFrame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::putText(placeholderFrame, "Camera not found.", cv::Point(50, 240),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(placeholderFrame, "Connect camera and reboot.", cv::Point(50, 300),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        while (g_running) {
            if (httpServer.isStreaming()) {
                std::vector<uchar> buff_bgr;
                cv::imencode(".jpg", placeholderFrame, buff_bgr, params);
                httpServer.streamFrame("/", std::string(buff_bgr.begin(), buff_bgr.end()));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    } else {
        cv::Mat frame, gray, decimated;
        while (g_running) {
            if (!cameraManager.read(frame) || frame.empty()) {
                cv::Mat emptyFrame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
                cv::putText(emptyFrame, "Frame empty.", cv::Point(50, 240),
                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                cv::putText(emptyFrame, "Reconnect camera and reboot", cv::Point(50, 300),
                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

                if (httpServer.isStreaming()) {
                    std::vector<uchar> buff_bgr;
                    cv::imencode(".jpg", emptyFrame, buff_bgr, params);
                    httpServer.streamFrame("/", std::string(buff_bgr.begin(), buff_bgr.end()));
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            cv::Mat detectGray = gray;
            if (decimateFactor > 1) {
                cv::resize(gray, decimated, cv::Size(), 1.0 / decimateFactor, 1.0 / decimateFactor,
                           cv::INTER_AREA);
                detectGray = decimated;
            }

            gpuDetector.dispatch(detectGray);

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

            for (const auto& det : tagDets) {
                if (!det.good) continue;
                TagDetectionData out;
                if (CpuPostProcessor::estimatePose(det, tagSizeM, cameraMatrix, distCoeffs, out)) {
                    result.detections.push_back(std::move(out));
                } else {
                    result.detections.push_back(CpuPostProcessor::toTagDetectionData(det));
                }
            }

            ntClient.publishDetections(result);
            sharedResults.update(std::move(result));

            if(httpServer.isStreaming()) [[likely]] {
                std::vector<uchar> buff_bgr;
                cv::imencode(".jpg", frame, buff_bgr, params);
                httpServer.streamFrame("/", std::string(buff_bgr.begin(), buff_bgr.end()));
            }

            if(httpServer.isCameraSettingsRefreshQueued()) [[unlikely]] {
                std::cout << "Refreshing camera settings from config...\n";
                cameraManager.applySettings(config);
                httpServer.clearCameraSettingsRefreshQueue();
            }

            if(httpServer.isCameraSnapshotQueued()) [[unlikely]] {
                std::string snapshotPath = opts.cameraSnapshotDir + "/image" + std::to_string(result.timestamp) + ".jpg";
                std::cout << "Saving snapshot to " << snapshotPath << std::endl;
                cv::imwrite(snapshotPath, frame);
                httpServer.clearCameraSnapshotQueue();
            }
        }
    }

    httpServer.stop();
    cameraManager.release();
    glContext.destroy();
    return 0;
}