#include "ocr_service.h"
#include "ocr.pb.h"

std::vector<ocr::TextResult> recognize_image(std::string image_path_expr);

grpc::Status OcrServiceImpl::recognize(
    grpc::ServerContext* context,
    const ocr::ImageRequest* request,
    ocr::OcrResponse* response) {

    const std::string& image = request->image();
    std::cout << "Start recognize" << std::endl;
    // TODO: Call your existing OCR implementation here.
    auto recognize_result = recognize_image("/app/WillekePass.jpg");
    for (const auto& result : recognize_result)
    {
    response->add_results()->CopyFrom(result);
    }

    return grpc::Status::OK;
}
