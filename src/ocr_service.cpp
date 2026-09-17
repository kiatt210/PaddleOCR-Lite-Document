#include "ocr_service.h"

grpc::Status OcrServiceImpl::recognize(
    grpc::ServerContext* context,
    const ocr::ImageRequest* request,
    ocr::OcrResponse* response) {

    const std::string& image = request->image();

    // TODO: Call your existing OCR implementation here.

    auto* result = response->add_results();

    result->set_text("Hello from OCR");
    result->set_confidence(0.98);

    auto* box = result->mutable_box();

    box->set_left(100);
    box->set_top(200);
    box->set_right(300);
    box->set_bottom(250);

    return grpc::Status::OK;
}
