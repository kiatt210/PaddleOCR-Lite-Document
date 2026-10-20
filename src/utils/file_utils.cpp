#include "utils/file_utils.h"

#include <fstream>
#include <limits>

namespace utils {

FileResult write_binary_file(
    const std::filesystem::path& path,
    std::string_view data)
{
    if (data.size() >
        static_cast<std::size_t>(
            std::numeric_limits<std::streamsize>::max())) {
        return {
            FileError::WriteFailed,
            std::make_error_code(std::errc::file_too_large)
        };
    }

    std::ofstream file;

    file.exceptions(std::ios::badbit);

    file.open(
        path,
        std::ios::binary | std::ios::out | std::ios::trunc);

    if (!file.is_open()) {
        return {
            FileError::OpenFailed,
            std::make_error_code(std::errc::io_error)
        };
    }

    file.write(
        data.data(),
        static_cast<std::streamsize>(data.size()));

    if (!file) {
        return {
            FileError::WriteFailed,
            std::make_error_code(std::errc::io_error)
        };
    }

    file.close();

    if (file.fail()) {
        return {
            FileError::CloseFailed,
            std::make_error_code(std::errc::io_error)
        };
    }

    return {};
}

}  // namespace utils
