#ifndef NETWORK_TABLES_HPP
#define NETWORK_TABLES_HPP
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include "ConfigManager.hpp"
#include "DetectionResult.hpp"
class NetworkTablesClient {
public:
    struct Config {
        bool enabled{false};
        std::string serverIp;
        int serverPort{5810};
        std::string clientName{"apriltag"};
    };
    enum class ConnectionState {
        Disabled,
        Connecting,
        Connected,
        Disconnected,
        Error
    };
    struct Status {
        ConnectionState state{ConnectionState::Disabled};
        std::string message;
        std::string serverAddress;
    };
    NetworkTablesClient();
    ~NetworkTablesClient();
    NetworkTablesClient(const NetworkTablesClient&) = delete;
    NetworkTablesClient& operator=(const NetworkTablesClient&) = delete;
    static Config configFromManager(const ConfigManager& config);
    static void applyConfigToManager(ConfigManager& config, const Config& ntConfig);
    void start(const Config& config);
    void stop();
    void updateConfig(const Config& config);
    void publishDetections(const DetectionFrame& frame);
    Status getStatus() const;
    bool isConnected() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#endif // NETWORK_TABLES_HPP