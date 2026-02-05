#ifndef LOGGER_H
#define LOGGER_H

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

namespace rtsp {

enum class CameraState {
  Connecting,
  Connected,
  StreamLost,
  Retrying,
  Reconnected,
  Disconnected,
  Error
};

enum class ErrorCategory {
  FrameCorruption,
  BitstreamError,
  DecoderInitFailed,
  HardwareAccelFailed,
  GpuFallback,
  NetworkError,
  Timeout,
  InvalidConfig,
  Unknown
};

inline std::string stateToString(CameraState state) {
  switch (state) {
    case CameraState::Connecting:   return "CONNECTING";
    case CameraState::Connected:    return "CONNECTED";
    case CameraState::StreamLost:   return "STREAM_LOST";
    case CameraState::Retrying:     return "RETRYING";
    case CameraState::Reconnected:  return "RECONNECTED";
    case CameraState::Disconnected: return "DISCONNECTED";
    case CameraState::Error:        return "ERROR";
    default:                        return "UNKNOWN";
  }
}

inline std::string categoryToString(ErrorCategory category) {
  switch (category) {
    case ErrorCategory::FrameCorruption:     return "FRAME_CORRUPTION";
    case ErrorCategory::BitstreamError:      return "BITSTREAM_ERROR";
    case ErrorCategory::DecoderInitFailed:   return "DECODER_INIT_FAILED";
    case ErrorCategory::HardwareAccelFailed: return "HARDWARE_ACCEL_FAILED";
    case ErrorCategory::GpuFallback:         return "GPU_FALLBACK";
    case ErrorCategory::NetworkError:        return "NETWORK_ERROR";
    case ErrorCategory::Timeout:             return "TIMEOUT";
    case ErrorCategory::InvalidConfig:       return "INVALID_CONFIG";
    case ErrorCategory::Unknown:             return "UNKNOWN";
    default:                                 return "UNKNOWN";
  }
}

class DateLogger {
public:
  DateLogger(const std::string& base_path, const std::string& camera_name)
      : base_path_(base_path), camera_name_(camera_name), current_date_("") {
    ensureLogDirectory();
  }

  ~DateLogger() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (log_file_.is_open()) {
      log_file_.close();
    }
  }

  DateLogger(const DateLogger&) = delete;
  DateLogger& operator=(const DateLogger&) = delete;

  // std::mutex is non-movable, so DateLogger must also be non-movable
  DateLogger(DateLogger&&) = delete;
  DateLogger& operator=(DateLogger&&) = delete;

  void logStateChange(CameraState new_state, const std::string& details = "") {
    std::ostringstream oss;
    oss << "[STATE_CHANGE] " << stateToString(new_state);
    if (!details.empty()) {
      oss << " - " << details;
    }
    writeLog("STATE", oss.str());
  }

  void logError(ErrorCategory category, const std::string& message,
                int error_code = 0, const std::string& affected_stream = "") {
    std::ostringstream oss;
    oss << "[ERROR] " << categoryToString(category) << " | " << message;
    if (error_code != 0) {
      oss << " | code=" << error_code;
    }
    if (!affected_stream.empty()) {
      oss << " | stream=" << affected_stream;
    }
    writeLog("ERROR", oss.str());
  }

  void logInfo(const std::string& message) {
    writeLog("INFO", message);
  }

  void logDebug(const std::string& message) {
    writeLog("DEBUG", message);
  }

  void logWarning(const std::string& message) {
    writeLog("WARN", message);
  }

  std::string getCurrentLogPath() const {
    std::lock_guard<std::mutex> lock(log_mutex_);
    return current_log_path_;
  }

private:
  std::string base_path_;
  std::string camera_name_;
  std::string current_date_;
  std::string current_log_path_;
  mutable std::mutex log_mutex_;
  std::ofstream log_file_;

  std::string getCurrentDateString() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time);
#else
    localtime_r(&time, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d");
    return oss.str();
  }

  std::string getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) % 1000;
    
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time);
#else
    localtime_r(&time, &tm_buf);
#endif
    
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
  }

  std::string getDateDirectoryPath() const {
    return base_path_ + "/" + getCurrentDateString();
  }

  std::string getLogFilePath() const {
    return getDateDirectoryPath() + "/" + camera_name_ + ".log";
  }

  void ensureLogDirectory() {
    std::string date_dir = getDateDirectoryPath();
    std::filesystem::create_directories(date_dir);
  }

  void checkDateRollover() {
    std::string new_date = getCurrentDateString();
    if (new_date != current_date_) {
      current_date_ = new_date;
      
      if (log_file_.is_open()) {
        log_file_.close();
      }
      
      ensureLogDirectory();
      current_log_path_ = getLogFilePath();
      log_file_.open(current_log_path_, std::ios::app);
    }
  }

  void writeLog(const std::string& level, const std::string& message) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    checkDateRollover();
    
    if (!log_file_.is_open()) {
      current_log_path_ = getLogFilePath();
      log_file_.open(current_log_path_, std::ios::app);
    }
    
    if (log_file_.is_open()) {
      log_file_ << "[" << getCurrentTimestamp() << "] "
                << "[" << camera_name_ << "] "
                << "[" << level << "] "
                << message << std::endl;
      log_file_.flush(); 
    }
  }
};

}  // namespace rtsp

#endif  // LOGGER_H
