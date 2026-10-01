#include "HttpServer.hpp"

#include <httplib.h>
#include <json.hpp>
#include "mjpeg_streamer.hpp"
#include <chrono>
#include <sstream>
#include "CameraCalibration.hpp"
#include <stdio.h>
#include <unistd.h>

using json = nlohmann::json;

struct HttpServer::ServerImpl {
    httplib::Server server;
};

std::string webuiDir = "webui"; // default webui directory

void HttpServer::setWebuiDir(const std::string& dir) {
    webuiDir = dir;
}

nadjieb::MJPEGStreamer mjpegStreamer;
int cameraStreamPort = 8081; // default camera stream port

void HttpServer::setCameraStreamPort(int port) {
    cameraStreamPort = port;
}

bool HttpServer::isStreaming() {
    return mjpegStreamer.isRunning();
}

void HttpServer::streamFrame(std::string url, std::string frameData) {
    if (mjpegStreamer.isRunning()) {
        mjpegStreamer.publish(url, std::move(frameData));
    }
}

bool queuedCameraSettingsRefresh = false;

bool HttpServer::isCameraSettingsRefreshQueued() {
    return queuedCameraSettingsRefresh;
}

void HttpServer::clearCameraSettingsRefreshQueue() {
    queuedCameraSettingsRefresh = false;
}

bool queuedCameraSnapshot = false;

bool HttpServer::isCameraSnapshotQueued() {
    return queuedCameraSnapshot;
}

void HttpServer::clearCameraSnapshotQueue() {
    queuedCameraSnapshot = false;
}

namespace {

double nowSeconds() {
    using clock = std::chrono::system_clock;
    auto now = clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}

json mat3ToJson(const cv::Mat& R) {
    json rows = json::array();
    for (int r = 0; r < 3; ++r) {
        json row = json::array();
        for (int c = 0; c < 3; ++c) {
            row.push_back(R.at<double>(r, c));
        }
        rows.push_back(row);
    }
    return rows;
}

} // anonymous namespace

HttpServer::HttpServer(int port, SharedResults& results, ConfigManager& config, CameraCalibration& calibration, NetworkTablesClient& ntClient)
    : port_(port), results_(results), config_(config), calibration_(calibration), ntClient_(ntClient), impl_(std::make_unique<ServerImpl>()) {}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::start() {
    if (running_.exchange(true)) return;

    auto& svr = impl_->server;

    svr.Get("/api/health", [](const httplib::Request&, httplib::Response& res) {
        json body;
        body["status"] = "ok";
        body["timestamp"] = nowSeconds();
        res.set_content(body.dump(), "application/json");
    });

    svr.Get("/api/detections", [this](const httplib::Request&, httplib::Response& res) {
        DetectionFrame frame = results_.getLatest();
        json body;
        body["timestamp"] = frame.timestamp;
        json tags = json::array();
        for (const auto& det : frame.detections) {
            json tag;
            tag["id"] = det.id;
            tag["hamming"] = det.hammingDist;
            tag["center"] = json::array({det.center.x, det.center.y});
            json corners = json::array();
            for (const auto& c : det.corners) {
                corners.push_back(json::array({c.x, c.y}));
            }
            tag["corners"] = corners;
            tags.push_back(tag);
        }
        body["tags"] = tags;
        res.set_content(body.dump(), "application/json");
    });

    svr.Get("/api/estimatedpose", [this](const httplib::Request&, httplib::Response& res) {
        DetectionFrame frame = results_.getLatest();
        json body;
        body["timestamp"] = frame.timestamp;
        json poses = json::array();
        for (const auto& det : frame.detections) {
            if (!det.hasPose) continue;
            json pose;
            pose["id"] = det.id;
            pose["rotation"] = mat3ToJson(det.rotationMatrix);
            pose["translation"] = json::array({
                det.translation[0], det.translation[1], det.translation[2]
            });
            poses.push_back(pose);
        }
        body["poses"] = poses;
        res.set_content(body.dump(), "application/json");
    });

    svr.Get("/", [this](const httplib::Request&, httplib::Response& res) {
        // return an HTML page loaded from the webuiDir_
        std::string indexPath = webuiDir + "/index.html";
        std::cout << webuiDir << std::endl;
        std::ifstream file(indexPath);
        if (!file.is_open()) {
            res.status = 404;
            res.set_content("Index file not found", "text/plain");
            return;
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        res.set_content(buffer.str(), "text/html");
    });

    svr.Get("/getconfig", [this](const httplib::Request&, httplib::Response& res) {
        json body;
        for (const auto& [key, value] : config_.getAll()) {
            body[key] = value;
        }
        res.set_content(body.dump(), "application/json");
    });

    svr.Get("/stream", [](const httplib::Request&, httplib::Response& res) {
        if (!mjpegStreamer.isRunning()) {
            mjpegStreamer.start(cameraStreamPort);
            res.set_content("MJPEG stream started", "text/plain");
        } else {
            res.set_content("MJPEG stream already running", "text/plain");
        }
        res.status = 200;
    });

    svr.Get("/stopstream", [](const httplib::Request&, httplib::Response& res) {
        if (mjpegStreamer.isRunning()) {
            mjpegStreamer.stop();
        }
        res.set_content("MJPEG stream stopped", "text/plain");
        res.status = 200;
    });

    svr.Get("/streamport", [](const httplib::Request&, httplib::Response& res) {
        json body;
        body["port"] = cameraStreamPort;
        res.set_content(body.dump(), "application/json");
        res.status = 200;
    });

    svr.Get("/queuecamerasettingsrefresh", [](const httplib::Request&, httplib::Response& res) {
        // return nothing, this is from the user updating camera settings
        queuedCameraSettingsRefresh = true;
        res.status = 200;
    });

    svr.Get("/takecamerasnapshot", [](const httplib::Request&, httplib::Response& res) {
        queuedCameraSnapshot = true;
        res.status = 200;
    });

    svr.Get("/clearsnapshots", [this](const httplib::Request&, httplib::Response& res) {
        this->calibration_.clearCalibrationImagePath();
        res.status = 200;
    });

    svr.Get("/calibratecamera", [this](const httplib::Request&, httplib::Response& res) {
        if(this->calibration_.calibrateCamera()){
            res.status = 200;
        } else {
            res.status = 500;
        }
    });

    svr.Get("/generatecharuco", [this](const httplib::Request&, httplib::Response& res) {
        cv::Mat charuco = this->calibration_.generateCharucoBoard();
        std::vector<uchar> buf;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 100};
        
        if (!cv::imencode(".jpg", charuco, buf, params)) {
            res.status = 500;
        } else {
            std::string body(reinterpret_cast<const char*>(buf.data()), buf.size());
            res.set_content(body, "image/jpeg");
            res.status = 200;
        }
    });

    svr.Get("/ntstatus", [this](const httplib::Request&, httplib::Response& res) {
        json body;
        NetworkTablesClient::Status status = this->ntClient_.getStatus();
        std::string stateStr;
        switch (status.state) {
            case NetworkTablesClient::ConnectionState::Disabled: stateStr = "Disabled"; break;
            case NetworkTablesClient::ConnectionState::Connecting: stateStr = "Connecting"; break;
            case NetworkTablesClient::ConnectionState::Connected: stateStr = "Connected"; break;
            case NetworkTablesClient::ConnectionState::Disconnected: stateStr = "Disconnected"; break;
            case NetworkTablesClient::ConnectionState::Error: stateStr = "Error"; break;
        }
        body["state"] = stateStr;
        body["message"] = status.message;
        body["serverAddress"] = status.serverAddress;
        res.set_content(body.dump(), "application/json");
    });

    svr.Get("/cpuusage", [](const httplib::Request&, httplib::Response& res) {
        json body;
        // Read CPU usage from /proc/stat
        std::ifstream statFile("/proc/stat");
        if (!statFile.is_open()) {
            res.status = 500;
            body["error"] = "Failed to open /proc/stat";
            res.set_content(body.dump(), "application/json");
            return;
        }

        std::string line;
        std::getline(statFile, line);
        std::istringstream iss(line);
        std::string cpuLabel;
        long user, nice, system, idle, iowait, irq, softirq, steal;
        iss >> cpuLabel >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

        long totalIdle = idle + iowait;
        long totalNonIdle = user + nice + system + irq + softirq + steal;
        long total = totalIdle + totalNonIdle;

        static long prevTotal = 0;
        static long prevIdle = 0;

        long totald = total - prevTotal;
        long idled = totalIdle - prevIdle;

        double cpuPercentage = (totald - idled) * 100.0 / totald;

        prevTotal = total;
        prevIdle = totalIdle;

        body["cpu_usage_percent"] = cpuPercentage;
        res.set_content(body.dump(), "application/json");
    });

    svr.Get("/memoryusage", [](const httplib::Request&, httplib::Response& res) {
        json body;
        // Read memory usage from /proc/meminfo
        std::ifstream memFile("/proc/meminfo");
        if (!memFile.is_open()) {
            res.status = 500;
            body["error"] = "Failed to open /proc/meminfo";
            res.set_content(body.dump(), "application/json");
            return;
        }

        std::string line;
        long memTotal = 0, memFree = 0, buffers = 0, cached = 0;

        while (std::getline(memFile, line)) {
            std::istringstream iss(line);
            std::string key;
            long value;
            std::string unit;
            iss >> key >> value >> unit;

            if (key == "MemTotal:") memTotal = value;
            else if (key == "MemFree:") memFree = value;
            else if (key == "Buffers:") buffers = value;
            else if (key == "Cached:") cached = value;

            if (memTotal && memFree && buffers && cached) break; // Stop if all values are found
        }

        long usedMemory = memTotal - memFree - buffers - cached;
        double usedMemoryPercent = static_cast<double>(usedMemory) / memTotal * 100.0;

        body["total_memory_kb"] = memTotal;
        body["used_memory_kb"] = usedMemory;
        body["used_memory_percent"] = usedMemoryPercent;

        res.set_content(body.dump(), "application/json");
    });

    svr.Post("/restartntclient", [this](const httplib::Request&, httplib::Response& res) {
        this->ntClient_.stop();
        this->ntClient_.start(this->ntClient_.configFromManager(this->config_));
        res.status = 200;
    });

    auto cameraCalibrationParametersHandler = [this](const httplib::Request& req, httplib::Response& res) {
        try {
            json body = json::parse(req.body);
            for (auto it = body.begin(); it != body.end(); ++it) {
                if(it.key() == "squaresX"){
                    this->calibration_.setBoardParametersSX(std::stoi(it.value().dump()));
                }
                if(it.key() == "squaresY"){
                    this->calibration_.setBoardParametersSY(std::stoi(it.value().dump()));
                }
                if(it.key() == "squareLength"){
                    this->calibration_.setBoardParametersSL(std::stof(it.value().dump()));
                }
                if(it.key() == "markerLength"){
                    this->calibration_.setBoardParametersML(std::stof(it.value().dump()));
                }
            }
            json ok;
            ok["status"] = "updated";
            res.set_content(ok.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            json err;
            err["error"] = e.what();
            res.set_content(err.dump(), "application/json");
        }
    };

    svr.Post("/setcalibrationparameters", cameraCalibrationParametersHandler);

    // handler for other files in the webuiDir
    svr.set_mount_point("/", webuiDir.c_str());

    auto configHandler = [this](const httplib::Request& req, httplib::Response& res) {
        try {
            json body = json::parse(req.body);
            for (auto it = body.begin(); it != body.end(); ++it) {
                if (it.value().is_string()) {
                    config_.setString(it.key(), it.value().get<std::string>());
                } else if (it.value().is_number_float()) {
                    config_.setDouble(it.key(), it.value().get<double>());
                } else if (it.value().is_number_integer()) {
                    config_.setInt(it.key(), it.value().get<int>());
                } else {
                    config_.setString(it.key(), it.value().dump());
                }
            }
            json ok;
            ok["status"] = "updated";
            res.set_content(ok.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            json err;
            err["error"] = e.what();
            res.set_content(err.dump(), "application/json");
        }
    };

    svr.Post("/api/config", configHandler);
    svr.Put("/api/config", configHandler);

    serverThread_ = std::thread([this]() {
        impl_->server.listen("0.0.0.0", port_);
        running_ = false;
    });
}

void HttpServer::stop() {
    if (!running_.exchange(false)) return;
    impl_->server.stop();
    if (serverThread_.joinable()) serverThread_.join();
}
