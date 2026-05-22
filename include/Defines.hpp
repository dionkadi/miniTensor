#pragma once

#include <cstdint>
#include <cstddef>
#include <string_view>
#include <type_traits>

constexpr size_t MAX_DIMS = 8;

enum class Dtype : uint8_t {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Int64   = 3,
    Int8    = 4
};

inline constexpr std::string_view dtype_name(Dtype d) {
    switch (d) {
        case Dtype::Float32: return "float32";
        case Dtype::Float16: return "float16";
        case Dtype::BFloat16: return "bfloat16";
        case Dtype::Int64:   return "int64";
        case Dtype::Int8:    return "int8";
    }
    return "unknown";
}

inline constexpr size_t dtype_size(Dtype d) {
    switch (d) {
        case Dtype::Float32: return 4;
        case Dtype::Float16: return 2;
        case Dtype::BFloat16: return 2;
        case Dtype::Int64:   return 8;
        case Dtype::Int8:    return 1;
    }
    return 0;
}

template<typename T>
inline constexpr Dtype dtype_of() {
    if constexpr (std::is_same_v<T, float>)       return Dtype::Float32;
    else if constexpr (std::is_same_v<T, int64_t>) return Dtype::Int64;
    else if constexpr (std::is_same_v<T, int8_t>)  return Dtype::Int8;
    // half and bfloat placeholder — add when stdlib types are available
    else static_assert(sizeof(T) == 0, "Unsupported dtype");
}

template<Dtype D>
struct dtype_t;

template<> struct dtype_t<Dtype::Float32> { using type = float; };
template<> struct dtype_t<Dtype::Int64>   { using type = int64_t; };
template<> struct dtype_t<Dtype::Int8>    { using type = int8_t; };
// half, bfloat: add when available

enum class DeviceType { CPU, CUDA };

struct Device {
    DeviceType type;
    uint8_t index;

    Device(DeviceType t = DeviceType::CPU, uint8_t i = 0): type(t), index(i) {}

    bool operator==(const Device& other) const {
        return type == other.type && index == other.index;
    }
};