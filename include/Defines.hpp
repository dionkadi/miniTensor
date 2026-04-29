#pragma once

#include <cstdint>
#include <cstddef>

constexpr size_t MAX_DIMS = 8;

enum class DeviceType { CPU, CUDA };

struct Device {
    DeviceType type;
    uint8_t index;

    Device(DeviceType t, uint8_t i = 0): type(t), index(i) {}

    bool operator==(const Device& other) const {
        return type == other.type && index == other.index;
    }
};