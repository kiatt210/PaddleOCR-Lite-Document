#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

namespace utils {

enum class FileError {
    None,
    EmptyData,
    OpenFailed,
    WriteFailed,
    CloseFailed
};

struct FileResult {
    FileError error = FileError::None;
    std::error_code system_error;

    [[nodiscard]] bool ok() const noexcept {
        return error == FileError::None;
    }

    explicit operator bool() const noexcept {
        return ok();
    }
};

// Writes binary data to a file, replacing its contents if it exists.
// Creates the file if it doesn't exist.
// Does not create missing parent directories.
[[nodiscard]] FileResult write_binary_file(
    const std::filesystem::path& path,
    std::string_view data);

}  // namespace utils
