#include <iostream>

#include <grpcpp/grpcpp.h>

#include "ocr_service.h"

int main() {
    const std::string address = "0.0.0.0:50051";

    OcrServiceImpl service;

    grpc::ServerBuilder builder;

    builder.AddListeningPort(
        address,
        grpc::InsecureServerCredentials());

    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();

    if (!server) {
        std::cerr << "Failed to start gRPC server\n";
        return 1;
    }

    std::cout << "OCR server listening on " << address << '\n';

    server->Wait();

    return 0;
}
