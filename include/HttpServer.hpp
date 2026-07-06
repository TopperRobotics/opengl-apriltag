#ifndef HTTP_SERVER_HPP
#define HTTP_SERVER_HPP

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "DetectionResult.hpp"
#include "ConfigManager.hpp"

class HttpServer {
public:
    HttpServer(int port, SharedResults& results, ConfigManager& config);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    void start();
    void stop();

    void setWebuiDir(const std::string& dir);
    void setCameraStreamPort(int port);
    bool isStreaming();
    void streamFrame(std::string url, std::string frameData);

    bool isCameraSettingsRefreshQueued();
    void clearCameraSettingsRefreshQueue();

private:
    int port_;
    SharedResults& results_;
    ConfigManager& config_;
    std::atomic<bool> running_{false};
    std::thread serverThread_;
    struct ServerImpl;
    std::unique_ptr<ServerImpl> impl_;
};

#endif // HTTP_SERVER_HPP
