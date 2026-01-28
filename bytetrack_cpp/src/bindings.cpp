#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "bytetrack/bytetrack.hpp"

namespace py = pybind11;

PYBIND11_MODULE(bytetrack_cpp, m) {
    m.doc() = "ByteTrack C++ implementation with Python bindings";
    
    // TrackState enum
    py::enum_<bytetrack::TrackState>(m, "TrackState")
        .value("New", bytetrack::TrackState::New)
        .value("Tracked", bytetrack::TrackState::Tracked)
        .value("Lost", bytetrack::TrackState::Lost)
        .value("Removed", bytetrack::TrackState::Removed)
        .export_values();
    
    // Detection struct
    py::class_<bytetrack::Detection>(m, "Detection")
        .def(py::init<>())
        .def(py::init<float, float, float, float, float, int, const std::string&>(),
             py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"),
             py::arg("confidence"), py::arg("class_id") = 0, py::arg("label") = "")
        .def_readwrite("x", &bytetrack::Detection::x)
        .def_readwrite("y", &bytetrack::Detection::y)
        .def_readwrite("width", &bytetrack::Detection::width)
        .def_readwrite("height", &bytetrack::Detection::height)
        .def_readwrite("confidence", &bytetrack::Detection::confidence)
        .def_readwrite("class_id", &bytetrack::Detection::class_id)
        .def_readwrite("label", &bytetrack::Detection::label);
    
    // TrackerConfig struct
    py::class_<bytetrack::TrackerConfig>(m, "TrackerConfig")
        .def(py::init<>())
        .def(py::init<float, float, int, int>(),
             py::arg("track_thresh"), py::arg("match_thresh"),
             py::arg("track_buffer"), py::arg("frame_rate") = 30)
        .def_readwrite("track_thresh", &bytetrack::TrackerConfig::track_thresh)
        .def_readwrite("high_thresh", &bytetrack::TrackerConfig::high_thresh)
        .def_readwrite("match_thresh", &bytetrack::TrackerConfig::match_thresh)
        .def_readwrite("track_buffer", &bytetrack::TrackerConfig::track_buffer)
        .def_readwrite("frame_rate", &bytetrack::TrackerConfig::frame_rate);
    
    // TrackInfo struct (for easy Python access)
    py::class_<bytetrack::TrackInfo>(m, "TrackInfo")
        .def(py::init<>())
        .def_readonly("track_id", &bytetrack::TrackInfo::track_id)
        .def_readonly("x", &bytetrack::TrackInfo::x)
        .def_readonly("y", &bytetrack::TrackInfo::y)
        .def_readonly("width", &bytetrack::TrackInfo::width)
        .def_readonly("height", &bytetrack::TrackInfo::height)
        .def_readonly("confidence", &bytetrack::TrackInfo::confidence)
        .def_readonly("class_id", &bytetrack::TrackInfo::class_id)
        .def_readonly("state", &bytetrack::TrackInfo::state)
        .def_property_readonly("tlwh", [](const bytetrack::TrackInfo& t) {
            return std::make_tuple(t.x, t.y, t.width, t.height);
        });
    
    // STrack class
    py::class_<bytetrack::STrack, std::shared_ptr<bytetrack::STrack>>(m, "STrack")
        .def_property_readonly("track_id", &bytetrack::STrack::trackId)
        .def_property_readonly("class_id", &bytetrack::STrack::classId)
        .def_property_readonly("score", &bytetrack::STrack::score)
        .def_property_readonly("state", &bytetrack::STrack::state)
        .def_property_readonly("is_activated", &bytetrack::STrack::isActivated)
        .def_property_readonly("tlwh", [](const bytetrack::STrack& t) {
            auto box = t.tlwh();
            return std::make_tuple(box(0), box(1), box(2), box(3));
        })
        .def_property_readonly("tlbr", [](const bytetrack::STrack& t) {
            auto box = t.tlbr();
            return std::make_tuple(box(0), box(1), box(2), box(3));
        });
    
    // BYTETracker class
    py::class_<bytetrack::BYTETracker>(m, "BYTETracker")
        .def(py::init<>())
        .def(py::init<const bytetrack::TrackerConfig&>(), py::arg("config"))
        .def("update", &bytetrack::BYTETracker::update, py::arg("detections"),
             "Update tracker with new detections")
        .def("get_track_info", &bytetrack::BYTETracker::getTrackInfo,
             "Get current track information")
        .def("reset", &bytetrack::BYTETracker::reset,
             "Reset tracker state")
        .def_property_readonly("frame_id", &bytetrack::BYTETracker::frameId);
    
    // Convenience function to create tracker with common parameters
    m.def("create_tracker", [](float track_thresh, float match_thresh, 
                               int track_buffer, int frame_rate) {
        bytetrack::TrackerConfig config(track_thresh, match_thresh, track_buffer, frame_rate);
        return bytetrack::BYTETracker(config);
    }, py::arg("track_thresh") = 0.5f,
       py::arg("match_thresh") = 0.8f,
       py::arg("track_buffer") = 30,
       py::arg("frame_rate") = 30,
       "Create a ByteTrack tracker with specified parameters");
}

