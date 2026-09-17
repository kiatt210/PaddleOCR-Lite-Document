#pragma once

#include <grpcpp/grpcpp.h>

#include "ocr.grpc.pb.h"

class OcrServiceImpl final : public ocr::OcrService::Service {
public:
    grpc::Status recognize(
        grpc::ServerContext* context,
        const ocr::ImageRequest* request,
        ocr::OcrResponse* response) override;
};
