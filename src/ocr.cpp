// Copyright (c) 2026 Q-engineering. All Rights Reserved.
// Modified for NCNN Engine (PP-OCRv5 support)

#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

// Replace Paddle headers with NCNN
#include "net.h"

#include "cls_process.h"
#include "crnn_process.h"
#include "db_post_process.h"
#include <memory>
#include <stdexcept>
#include <sstream>
#include "ocr.pb.h"

using namespace std;
const string runtime_device = "arm8";

// resize image to a size multiple of 32 which is required by the network
cv::Mat DetResizeImg(const cv::Mat img, int max_size_len, std::vector<float> &ratio_hw) {
  int w = img.cols;
  int h = img.rows;

  float ratio = 1.f;
  int max_wh = w >= h ? w : h;
  if (max_wh > max_size_len) {
    if (h > w) {
      ratio = static_cast<float>(max_size_len) / static_cast<float>(h);
    } else {
      ratio = static_cast<float>(max_size_len) / static_cast<float>(w);
    }
  }

  int resize_h = static_cast<int>(float(h) * ratio);
  int resize_w = static_cast<int>(float(w) * ratio);
  if (resize_h % 32 == 0) resize_h = resize_h;
  else if (resize_h / 32 < 1 + 1e-5) resize_h = 32;
  else resize_h = (resize_h / 32 - 1) * 32;

  if (resize_w % 32 == 0) resize_w = resize_w;
  else if (resize_w / 32 < 1 + 1e-5) resize_w = 32;
  else resize_w = (resize_w / 32 - 1) * 32;

  cv::Mat resize_img;
  cv::resize(img, resize_img, cv::Size(resize_w, resize_h));

  ratio_hw.push_back(static_cast<float>(resize_h) / static_cast<float>(h));
  ratio_hw.push_back(static_cast<float>(resize_w) / static_cast<float>(w));
  return resize_img;
}

cv::Mat RunClsModel(cv::Mat img, ncnn::Net* predictor_cls, const float thresh = 0.9) {
  cv::Mat srcimg;
  img.copyTo(srcimg);
  cv::Mat crop_img;
  img.copyTo(crop_img);

  cv::Mat resize_img = ClsResizeImg(crop_img);

  // NCNN handles normalization natively, no need for NeonMeanScale
  ncnn::Mat in = ncnn::Mat::from_pixels(resize_img.data, ncnn::Mat::PIXEL_BGR2RGB, resize_img.cols, resize_img.rows);
  const float mean_vals[3] = {0.5f * 255.f, 0.5f * 255.f, 0.5f * 255.f};
  const float norm_vals[3] = {1 / 0.5f / 255.f, 1 / 0.5f / 255.f, 1 / 0.5f / 255.f};
  in.substract_mean_normalize(mean_vals, norm_vals);

  ncnn::Extractor ex = predictor_cls->create_extractor();
  ex.input("input", in);

  ncnn::Mat out;
  ex.extract("out", out);

  float score = 0;
  int label = 0;
  for (int i = 0; i < out.w; i++) {
    if (out[i] > score) {
      score = out[i];
      label = i;
    }
  }

  if (label % 2 == 1 && score > thresh) {
    cv::rotate(srcimg, srcimg, 1);
  }
  return srcimg;
}

void RunRecModel(std::vector<std::vector<std::vector<int>>> boxes, cv::Mat img,
                 ncnn::Net* predictor_crnn,
                 std::vector<std::string> &rec_text,
                 std::vector<float> &rec_text_score,
                 std::vector<std::string> charactor_dict,
                 ncnn::Net* predictor_cls,
                 int use_direction_classify,
                 std::vector<double> *times,
                 int rec_image_height) {
  cv::Mat srcimg;
  img.copyTo(srcimg);
  cv::Mat crop_img;
  cv::Mat resize_img;

  std::vector<double> time_info = {0, 0, 0};

  // NCNN norm values for Rec
  const float mean_vals[3] = {0.5f * 255.f, 0.5f * 255.f, 0.5f * 255.f};
  const float norm_vals[3] = {1 / 0.5f / 255.f, 1 / 0.5f / 255.f, 1 / 0.5f / 255.f};

  for (int i = 0; i < boxes.size(); i++) {
    auto preprocess_start = std::chrono::steady_clock::now();
    crop_img = GetRotateCropImage(srcimg, boxes[i]);
    if (use_direction_classify >= 1 && predictor_cls != nullptr) {
      crop_img = RunClsModel(crop_img, predictor_cls);
    }
    float wh_ratio = static_cast<float>(crop_img.cols) / static_cast<float>(crop_img.rows);

    rec_image_height = 48;
    resize_img = CrnnResizeImg(crop_img, wh_ratio, rec_image_height);

    ncnn::Mat in = ncnn::Mat::from_pixels(resize_img.data, ncnn::Mat::PIXEL_BGR2RGB, resize_img.cols, resize_img.rows);
    in.substract_mean_normalize(mean_vals, norm_vals);
    auto preprocess_end = std::chrono::steady_clock::now();

    //// Run CRNN predictor
    auto inference_start = std::chrono::steady_clock::now();
    ncnn::Extractor ex = predictor_crnn->create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);
    auto inference_end = std::chrono::steady_clock::now();

    // ctc decode
    auto postprocess_start = std::chrono::steady_clock::now();
    std::string str_res;
    int argmax_idx;
    int last_index = 0;
    float score = 0.f;
    int count = 0;
    float max_value = 0.0f;

    // NCNN outputs CRNN as (w=num_classes, h=seq_len)
    for (int n = 0; n < out.h; n++) {
      const float* row_ptr = out.row(n);
      argmax_idx = int(Argmax(row_ptr, row_ptr + out.w));
      max_value = float(*std::max_element(row_ptr, row_ptr + out.w));

      if (argmax_idx > 0 && argmax_idx < charactor_dict.size()){
        if(!(n > 0 && argmax_idx == last_index)) {
            score += max_value;
            count += 1;
            str_res += charactor_dict[argmax_idx];
          }
        }
        last_index = argmax_idx;
    }

    if (count > 0) score /= count;
    rec_text.push_back(str_res);
    rec_text_score.push_back(score);
    auto postprocess_end = std::chrono::steady_clock::now();

    std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
    time_info[0] += double(preprocess_diff.count() * 1000);
    std::chrono::duration<float> inference_diff = inference_end - inference_start;
    time_info[1] += double(inference_diff.count() * 1000);
    std::chrono::duration<float> postprocess_diff = postprocess_end - postprocess_start;
    time_info[2] += double(postprocess_diff.count() * 1000);
  }

  times->push_back(time_info[0]);
  times->push_back(time_info[1]);
  times->push_back(time_info[2]);
}

std::vector<std::vector<std::vector<int>>> RunDetModel(ncnn::Net* predictor, cv::Mat img, std::map<std::string, double> Config, std::vector<double> *times) {
  int max_side_len = int(Config["max_side_len"]);
  int det_db_use_dilate = int(Config["det_db_use_dilate"]);

  cv::Mat srcimg;
  img.copyTo(srcimg);

  auto preprocess_start = std::chrono::steady_clock::now();
  std::vector<float> ratio_hw;
  img = DetResizeImg(img, max_side_len, ratio_hw);

  // Prepare NCNN input
  ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);
  const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
  const float norm_vals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};
  in.substract_mean_normalize(mean_vals, norm_vals);
  auto preprocess_end = std::chrono::steady_clock::now();

  // Run predictor
  auto inference_start = std::chrono::steady_clock::now();
  ncnn::Extractor ex = predictor->create_extractor();
  ex.input("in0", in);
  ncnn::Mat out;
  ex.extract("out0", out);
  auto inference_end = std::chrono::steady_clock::now();

  // Save output (out.w is width, out.h is height)
  auto postprocess_start = std::chrono::steady_clock::now();

  std::vector<float> pred(out.w * out.h);
  std::vector<unsigned char> cbuf(out.w * out.h);

  const float* outptr = out.row(0); // contiguous float array for c=1

  for (int i = 0; i < out.w * out.h; i++) {
    pred[i] = static_cast<float>(outptr[i]);
    cbuf[i] = static_cast<unsigned char>((outptr[i]) * 255);
  }

  cv::Mat cbuf_map(out.h, out.w, CV_8UC1, reinterpret_cast<unsigned char *>(cbuf.data()));
  cv::Mat pred_map(out.h, out.w, CV_32F, reinterpret_cast<float *>(pred.data()));

  const double threshold = double(Config["det_db_thresh"]) * 255;
  const double max_value = 255;
  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, max_value, cv::THRESH_BINARY);

  if (det_db_use_dilate == 1) {
    cv::Mat dilation_map;
    cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(bit_map, dilation_map, dila_ele);
    bit_map = dilation_map;
  }

  auto boxes = BoxesFromBitmap(pred_map, bit_map, Config);
  std::vector<std::vector<std::vector<int>>> filter_boxes = FilterTagDetRes(boxes, ratio_hw[0], ratio_hw[1], srcimg);
  auto postprocess_end = std::chrono::steady_clock::now();

  std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
  times->push_back(double(preprocess_diff.count() * 1000));
  std::chrono::duration<float> inference_diff = inference_end - inference_start;
  times->push_back(double(inference_diff.count() * 1000));
  std::chrono::duration<float> postprocess_diff = postprocess_end - postprocess_start;
  times->push_back(double(postprocess_diff.count() * 1000));

  return filter_boxes;
}

ncnn::Net* loadModel(const std::string& model_file, int num_threads) {
  auto net = std::make_unique<ncnn::Net>();
  net->opt.use_vulkan_compute = false;
  net->opt.use_fp16_arithmetic = true; // HUGE speedup on Pi
  net->opt.num_threads = num_threads;

  if (net->load_param((model_file + ".param").c_str()) != 0 ||
      net->load_model((model_file + ".bin").c_str()) != 0) {
      throw std::runtime_error("NCNN model loading failed: " + model_file);
  }
  return net.release();
}

std::vector<std::string> split(const std::string &str, const std::string &delim) {
  std::vector<std::string> res;
  if (str.empty() || delim.empty()) {
    return res;
  }

  std::size_t start = 0;
  while (start < str.size()) {
    const auto end = str.find(delim, start);
    if (end == std::string::npos) {
      res.push_back(str.substr(start));
      break;
    }

    res.push_back(str.substr(start, end - start));
    start = end + delim.size();
  }

  return res;
}

std::map<std::string, double> LoadConfigTxt(std::string config_path) {
  auto config = ReadDict(config_path);
  std::map<std::string, double> dict;
  for (const auto& line : config) {
    std::istringstream stream(line);
    std::string key;
    double value;

    if (stream >> key >> value) {
      dict[key] = value;
    }
  }

  return dict;
}

cv::Mat Visualization(cv::Mat srcimg, std::vector<std::vector<std::vector<int>>> boxes) {
  cv::Mat img_vis;
  srcimg.copyTo(img_vis);
  for (int n = 0; n < boxes.size(); n++) {
    std::vector<cv::Point> pts;
    for (int m = 0; m < boxes[0].size(); m++) {
      pts.push_back(cv::Point(boxes[n][m][0], boxes[n][m][1]));
    }
    const cv::Point* ppt[1] = { pts.data() };
    int npt[] = { 4 };
    cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
  }

  cv::imwrite("./vis.jpg", img_vis);
  std::cout << "The detection visualized image saved in ./vis.jpg" << std::endl;
  return img_vis;
}

void convertToBoundingBox(
    const std::vector<std::vector<int>>& points, ocr::TextResult& result)
{

    if (points.empty())
        return;

    int left = points[0][0];
    int right = points[0][0];
    int top = points[0][1];
    int bottom = points[0][1];

    for (const auto& point : points)
    {
        if (point.size() < 2)
            continue;

        left = std::min(left, point[0]);
        right = std::max(right, point[0]);
        top = std::max(top, point[1]);
        bottom = std::min(bottom, point[1]);
    }

    ocr::BoundingBox* box = result.mutable_box();
    box->set_left(left);
    box->set_top(top);
    box->set_right(right);
    box->set_bottom(bottom);

}

class OcrEngine {
public:
  OcrEngine()
      : config_(LoadConfigTxt("/app/models/config.txt")),
        use_direction_classify_(
            static_cast<int>(config_["use_direction_classify"])),
        rec_image_height_(
            static_cast<int>(config_["rec_image_height"])),
        det_predictor_(
            loadModel("/app/models/PP_OCRv5_mobile_det", 4)),
        rec_predictor_(
            loadModel("/app/models/PP_OCRv5_mobile_rec", 4)) {
    charactor_dict_ = ReadDict("/app/models/PP_OCRv5_vocab.txt");
    charactor_dict_.insert(charactor_dict_.begin(), "#");
    charactor_dict_.push_back(" ");

    if (use_direction_classify_ >= 1) {
      cls_predictor_.reset(
          loadModel("/app/models/cls-sim-op", 4));
    }
  }

  std::vector<ocr::TextResult> recognize(
      const std::string& image_path_expr) {
    std::vector<ocr::TextResult> results;
    std::vector<cv::String> cv_all_img_names;
    cv::glob(image_path_expr, cv_all_img_names);

    std::vector<double> det_time_info = {0, 0, 0};
    std::vector<double> rec_time_info = {0, 0, 0};
    for (const auto& image_name : cv_all_img_names) {
      cv::Mat srcimg = cv::imread(image_name, cv::IMREAD_COLOR);
      if (srcimg.empty()) {
        continue;
      }

    std::vector<double> det_times;
      auto boxes = RunDetModel(
          det_predictor_.get(), srcimg, config_, &det_times);
 
      std::vector<std::string> rec_text;
      std::vector<float> rec_text_score;
      std::vector<double> rec_times;


      RunRecModel(
          boxes,
          srcimg,
          rec_predictor_.get(),
          rec_text,
          rec_text_score,
          charactor_dict_,
          cls_predictor_.get(),
          use_direction_classify_,
          &rec_times,
          rec_image_height_);
  
      for (std::size_t i = 0; i < rec_text.size(); ++i) {
        ocr::TextResult result;
        result.set_text(rec_text[i]);
        result.set_confidence(rec_text_score[i]);
        convertToBoundingBox(boxes[i], result);
        results.push_back(result);
      }

      for (std::size_t i = 0; i < 3; ++i) {
        det_time_info[i] += det_times[i];
        rec_time_info[i] += rec_times[i];
      }
    }

    return results;
  }

private:
  std::map<std::string, double> config_;
  int use_direction_classify_;
  int rec_image_height_;
  std::vector<std::string> charactor_dict_;
  std::unique_ptr<ncnn::Net> det_predictor_;
  std::unique_ptr<ncnn::Net> rec_predictor_;
  std::unique_ptr<ncnn::Net> cls_predictor_;
};

std::vector<ocr::TextResult> recognize_image(
    const std::string& image_path_expr) {
  static OcrEngine engine;
  return engine.recognize(image_path_expr);
}
