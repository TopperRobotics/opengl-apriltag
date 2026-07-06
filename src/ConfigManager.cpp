#include "ConfigManager.hpp"
#include <sqlite3.h>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <map>

const char* const ConfigManager::DEFAULTS[] = {
    "camera_fx",             "800",
    "camera_fy",             "800",
    "camera_cx",             "640",
    "camera_cy",             "360",
    "dist_coeffs",           "[0.1,-0.2,0,0,0]",
    "tag_size_m",            "0.16",
    "adaptive_threshold_win","31",
    "adaptive_threshold_const","7",
    "min_tag_area",          "100",
    "max_tag_area",          "10000",
    "decimate_factor",       "2",
    "camera_exposure",       "-6",
    "camera_brightness",     "0.5",
    "camera_autoexposure",   "0.75"
};

static const char* sqlCreateTable =
    "CREATE TABLE IF NOT EXISTS config ("
    "  key   TEXT PRIMARY KEY,"
    "  value TEXT NOT NULL"
    ")";

ConfigManager::ConfigManager()
    : map_(nullptr), db_(nullptr) {}

ConfigManager::~ConfigManager() {
    if (map_) delete map_;
    if (db_) sqlite3_close(db_);
}

bool ConfigManager::initDefaults(sqlite3* db) {
    std::map<std::string, bool> inserted;
    for (size_t i = 0; i < NUM_DEFAULTS; i += 2) {
        auto r = inserted.emplace(DEFAULTS[i], false);
        if (!r.second) continue; // duplicate key

        std::string sql = "INSERT OR IGNORE INTO config VALUES ('" +
                          std::string(DEFAULTS[i]) + "', '" +
                          std::string(DEFAULTS[i + 1]) + "')";
        char* err{nullptr};
        int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err);
        if (rc != SQLITE_OK) {
            // ignore duplicates; other errors are bad
            if (err) sqlite3_free(err);
        }
    }
    return true;
}

bool ConfigManager::open(const std::string& dbPath) {
    std::lock_guard<std::mutex> lock(mutex_);

    int rc = sqlite3_open(dbPath.c_str(), &db_);
    if (rc != SQLITE_OK) {
        return false;
    }

    // Enable WAL mode for concurrent reads during HTTP requests
    const char* wal = "PRAGMA journal_mode=WAL;";
    sqlite3_exec(db_, wal, nullptr, nullptr, nullptr);

    // Load config rows from DB into a map (thread-safe copy-on-write)
    rc = sqlite3_exec(db_, sqlCreateTable, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK && db_) {
        return false;
    }

    initDefaults(db_);

    std::map<std::string, std::string> tmpMap;
    const char* sql = "SELECT key, value FROM config ORDER BY key";
    sqlite3_stmt* stmt{nullptr};
    rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc == SQLITE_OK) {
        while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
            const char* k = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            const char* v = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if (k && v) tmpMap[std::string(k)] = std::string(v);
        }
    }
    sqlite3_finalize(stmt);

    // Swap into map atomically (map pointer swap)
    auto* old = map_;
    map_ = new std::unordered_map<std::string, std::string>(tmpMap.begin(), tmpMap.end());
    if (old) delete old;

    dbPath_ = dbPath;
    return true;
}

float  ConfigManager::getFloat(const std::string& key, float def) const {
    auto* m{map_};
    if (!m) return def;
    auto it = m->find(key);
    if (it != m->end()) return std::stof(it->second);
    return def;
}
int ConfigManager::getInt(const std::string& key, int def) const {
    auto* m{map_};
    if (!m) return def;
    auto it = m->find(key);
    if (it != m->end()) return std::stoi(it->second);
    return def;
}
double ConfigManager::getDouble(const std::string& key, double def) const {
    auto* m{map_};
    if (!m) return def;
    auto it = m->find(key);
    if (it != m->end()) return std::stod(it->second);
    return def;
}
std::string ConfigManager::getString(const std::string& key, const std::string& def) const {
    auto* m{map_};
    if (!m) return def;
    auto it = m->find(key);
    if (it != m->end()) return it->second;
    return def;
}
bool ConfigManager::getBool(const std::string& key, bool def) const {
    auto* m{map_};
    if (!m) return def;
    auto it = m->find(key);
    if (it != m->end()) {
        auto v = it->second;
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        return v == "1" || v == "true";
    }
    return def;
}

auto ConfigManager::getAll() const -> std::unordered_map<std::string, std::string> {
    auto* m{map_};
    if (!m) return {};
    return *m;
}

void ConfigManager::setFloat(const std::string& key, float val) { setString(key, std::to_string(val)); }
void ConfigManager::setInt   (const std::string& key, int val)   { setString(key, std::to_string(val)); }
void ConfigManager::setDouble(const std::string& key, double val){ setString(key, std::to_string(val)); }

void ConfigManager::setString(const std::string& key, const std::string& val) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (map_) (*map_)[key] = val;

    // Write to DB
    if (db_) {
        std::string sql = "INSERT OR REPLACE INTO config VALUES ('" + key + "', '" + val + "')";
        sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, nullptr);
    }
}

void ConfigManager::reload() {
    if (!db_) return;
    const char* sql = "SELECT key, value FROM config ORDER BY key";
    sqlite3_stmt* stmt{nullptr};
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK || !map_) {
        sqlite3_finalize(stmt);
        return;
    }
    std::unordered_map<std::string, std::string> tmpMap;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const char* k = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        const char* v = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        if (k && v) tmpMap[std::string(k)] = std::string(v);
    }
    sqlite3_finalize(stmt);

    auto* old = map_;
    map_ = new std::unordered_map<std::string, std::string>(tmpMap.begin(), tmpMap.end());
    if (old) delete old;
}
