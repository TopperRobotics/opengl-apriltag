#ifndef CONFIG_MANAGER_HPP
#define CONFIG_MANAGER_HPP

#include <string>
#include <unordered_map>
#include <mutex>

struct sqlite3;

class ConfigManager {
public:
    static const char* const DEFAULTS[];
    static constexpr size_t NUM_DEFAULTS = 10*2;

    ConfigManager();
    ~ConfigManager();

    // Open database, run init-sql if empty. Blocking on open.
    bool open(const std::string& dbPath);

    // Typed getters (returns default if key not present)
    float  getFloat(const std::string& key, float def = 0.f) const;
    int    getInt(const std::string& key, int def = 0) const;
    double getDouble(const std::string& key, double def = 0.0) const;
    std::string getString(const std::string& key, const std::string& def = "") const;
    bool     getBool(const std::string& key, bool def = false) const;
    std::unordered_map<std::string, std::string> getAll() const;

    // Setter (writes to DB as well)
    void setFloat(const std::string& key, float val);
    void setInt(const std::string& key, int val);
    void setDouble(const std::string& key, double val);
    void setString(const std::string& key, const std::string& val);

    // Reload cache from DB (thread-safe)
    void reload();

    // Check if config is loaded
    bool isOpen() const { return dbPath_.size() > 0; }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::string>* map_{nullptr};
    sqlite3* db_{nullptr};
    std::string dbPath_;

    // internal helpers
    static bool initDefaults(sqlite3* db);
};

#endif
