#include "NetworkTables.hpp"
#include "DetectionResult.hpp"
#include "ConfigManager.hpp"

#include <networktables/NetworkTableInstance.h>
#include <networktables/NetworkTable.h>
#include <networktables/StringTopic.h>

#include <sstream>
#include <iomanip>
#include <cmath>
#include <mutex>

namespace {

// Helper: append a double with full precision and handle NaN/Infinity
void appendDouble(std::ostringstream& oss, double v) {
    if (std::isnan(v)) {
        oss << "null";
    } else if (std::isinf(v)) {
        oss << (v > 0 ? "\"Infinity\"" : "\"-Infinity\"");
    } else {
        oss << std::fixed << std::setprecision(6) << v;
    }
}

// Serialise a DetectionFrame to a JSON string for NetworkTables publishing
std::string toJson(const DetectionFrame& frame) {
    std::ostringstream oss;
    oss << "{\"timestamp\":";
    appendDouble(oss, frame.timestamp);

    oss << ",\"detections\":[";
    bool firstDet = true;
    for (const auto& det : frame.detections) {
        if (!firstDet) oss << ",";
        firstDet = false;
        oss << "{";
        oss << "\"id\":" << det.id;
        oss << ",\"hamming\":" << det.hammingDist;

        // center
        oss << ",\"center\":[";
        appendDouble(oss, det.center.x); oss << ",";
        appendDouble(oss, det.center.y); oss << "]";

        // corners (4 points, each [x,y])
        oss << ",\"corners\":[";
        for (size_t i = 0; i < det.corners.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "[";
            appendDouble(oss, det.corners[i].x); oss << ",";
            appendDouble(oss, det.corners[i].y); oss << "]";
        }
        oss << "]";

        // pose
        oss << ",\"hasPose\":" << (det.hasPose ? "true" : "false");
        if (det.hasPose) {
            // translation
            oss << ",\"translation\":[";
            appendDouble(oss, det.translation[0]); oss << ",";
            appendDouble(oss, det.translation[1]); oss << ",";
            appendDouble(oss, det.translation[2]); oss << "]";

            // rotation matrix (3x3) flattened row-major
            oss << ",\"rotation\":[";
            bool firstElem = true;
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    if (!firstElem) oss << ",";
                    firstElem = false;
                    appendDouble(oss, det.rotationMatrix.at<double>(r, c));
                }
            }
            oss << "]";
        }
        oss << "}";
    }
    oss << "]}";
    return oss.str();
}

} // anonymous namespace

// ---------- Impl ----------
struct NetworkTablesClient::Impl {
    Config config;
    nt::NetworkTableInstance inst;
    nt::NetworkTableEntry jsonEntry;
    mutable std::mutex mutex;
    bool started = false;
};

// ---------- public methods ----------
NetworkTablesClient::NetworkTablesClient()
    : impl_(std::make_unique<Impl>()) {}

NetworkTablesClient::~NetworkTablesClient() {
    stop();
}

NetworkTablesClient::Config NetworkTablesClient::configFromManager(const ConfigManager& config) {
    Config c;
    c.enabled    = config.getBool("NetworkTablesEnabled", true);
    c.serverIp   = config.getString("NetworkTablesServerIP", "");
    c.serverPort = config.getInt("NetworkTablesServerPort", 5810);
    c.clientName = config.getString("NetworkTablesClientName", "apriltag");
    return c;
}

void NetworkTablesClient::applyConfigToManager(ConfigManager& config, const Config& ntConfig) {
    // ConfigManager does not have setBool; store as "true"/"false"
    config.setString("NetworkTablesEnabled", ntConfig.enabled ? "true" : "false");
    config.setString("NetworkTablesServerIP", ntConfig.serverIp);
    config.setInt("NetworkTablesServerPort", ntConfig.serverPort);
    config.setString("NetworkTablesClientName", ntConfig.clientName);
}

void NetworkTablesClient::start(const Config& config) {
    std::lock_guard lock(impl_->mutex);

    // Stop any previous instance cleanly
    if (impl_->started) {
        impl_->inst.StopClient();
        impl_->inst = nt::NetworkTableInstance{};
        impl_->jsonEntry = nt::NetworkTableEntry{};
        impl_->started = false;
    }

    if (!config.enabled) {
        impl_->config = config;
        return;
    }

    impl_->config = config;
    impl_->inst = nt::NetworkTableInstance::GetDefault();
    impl_->inst.SetServer(config.serverIp.c_str(), config.serverPort);
    impl_->inst.StartClient4(config.clientName);

    auto table = impl_->inst.GetTable(config.clientName);
    impl_->jsonEntry = table->GetEntry("json");
    impl_->jsonEntry.SetString("{}");
    impl_->started = true;
}

void NetworkTablesClient::stop() {
    std::lock_guard lock(impl_->mutex);
    if (impl_->started) {
        impl_->inst.StopClient();
        impl_->inst = nt::NetworkTableInstance{};
        impl_->jsonEntry = nt::NetworkTableEntry{};
        impl_->started = false;
    }
    impl_->config.enabled = false;
}

void NetworkTablesClient::updateConfig(const Config& config) {
    // Simple restart-on-config-change approach
    start(config);
}

void NetworkTablesClient::publishDetections(const DetectionFrame& frame) {
    std::lock_guard lock(impl_->mutex);
    if (!impl_->started || !impl_->config.enabled) return;
    impl_->jsonEntry.SetString(toJson(frame));
}

NetworkTablesClient::Status NetworkTablesClient::getStatus() const {
    std::lock_guard lock(impl_->mutex);
    Status s;
    if (!impl_->config.enabled) {
        s.state = ConnectionState::Disabled;
        s.message = "NetworkTables is disabled";
        return s;
    }

    if (!impl_->started) {
        s.state = ConnectionState::Disconnected;
        s.message = "Not started";
        return s;
    }

    s.serverAddress = impl_->config.serverIp + ":" + std::to_string(impl_->config.serverPort);
    if (impl_->inst.IsConnected()) {
        s.state = ConnectionState::Connected;
        s.message = "Connected";
    } else {
        // Client started but not yet connected → treat as Connecting
        s.state = ConnectionState::Connecting;
        s.message = "Connecting...";
    }
    return s;
}

bool NetworkTablesClient::isConnected() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->started && impl_->config.enabled && impl_->inst.IsConnected();
}