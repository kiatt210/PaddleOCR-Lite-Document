#include "ocr_service.h"
#include "ocr.pb.h"
#include "ocr.h"
#include "utils/file_utils.h"
#include <iostream>
#include <exception>

grpc::Status OcrServiceImpl::recognize(
    grpc::ServerContext* context,
    const ocr::ImageRequest* request,
    ocr::OcrResponse* response) {

    std::cout << "Start recognize" << std::endl;
    
     try
    {
       const std::string& image = request->image();

       if (image.empty()) {
           return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,"Image data is empty");
       }

       const std::filesystem::path output_path = "/tmp/uploaded_image.jpg";

       const auto result = utils::write_binary_file(output_path, image);

       if (!result) {
           // Log result.error and result.system_error on the server.
	   return grpc::Status( grpc::StatusCode::INTERNAL,"Failed to save image");
       }
       auto recognize_result = recognize_image(output_path.string());
       std::cerr << "[DEBUG] After OCR" << std::endl;
       for (const auto& result : recognize_result)
       {
           response->add_results()->CopyFrom(result);
       }
       std::cerr << "[DEBUG] After create result" << std::endl;
       return grpc::Status::OK;
    }
    catch (const std::exception& e)
    {
        std::cerr << "OCR exception: " << e.what() << std::endl;

        return grpc::Status(
            grpc::StatusCode::INTERNAL,
            "OCR processing failed");
    }
    catch (...)
    {
        std::cerr << "OCR exception: unknown exception" << std::endl;

        return grpc::Status(
            grpc::StatusCode::INTERNAL,
            "Unexpected OCR processing error");
    }
}
