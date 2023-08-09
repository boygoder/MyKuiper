//
// Created by fss on 23-3-3.
//
#include "../source/layer/details/softmax.hpp"
#include "data/load_data.hpp"
#include "runtime/runtime_ir.hpp"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

// python ref https://pytorch.org/hub/pytorch_vision_resnet/
kuiper_infer::sftensor PreProcessImage(const cv::Mat &image) {
  using namespace kuiper_infer;
  assert(!image.empty());
  // 调整输入大小
  cv::Mat resize_image;
  cv::resize(image, resize_image, cv::Size(224, 224));

  cv::Mat rgb_image;
  cv::cvtColor(resize_image, rgb_image, cv::COLOR_BGR2RGB);

  rgb_image.convertTo(rgb_image, CV_32FC3);
  std::vector<cv::Mat> split_images;
  cv::split(rgb_image, split_images);
  uint32_t input_w = 224;
  uint32_t input_h = 224;
  uint32_t input_c = 3;
  sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);

  uint32_t index = 0;
  for (const auto &split_image : split_images) {
    assert(split_image.total() == input_w * input_h);
    const cv::Mat &split_image_t = split_image.t();
    memcpy(input->slice(index).memptr(), split_image_t.data,
           sizeof(float) * split_image.total());
    index += 1;
  }

  float mean_r = 0.485f;
  float mean_g = 0.456f;
  float mean_b = 0.406f;

  float var_r = 0.229f;
  float var_g = 0.224f;
  float var_b = 0.225f;
  assert(input->channels() == 3);
  input->data() = input->data() / 255.f;
  input->slice(0) = (input->slice(0) - mean_r) / var_r;
  input->slice(1) = (input->slice(1) - mean_g) / var_g;
  input->slice(2) = (input->slice(2) - mean_b) / var_b;
  return input;
}

TEST(test_model, resnet) {
  using namespace kuiper_infer;
  const std::string &param_path = "../tmp/resnet18_batch1.pnnx.param";
  const std::string &weight_path = "../tmp/resnet18_batch1.pnnx.bin";
  RuntimeGraph graph(param_path, weight_path);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  LOG(INFO) << "Start kuiperInfer inference";
  std::shared_ptr<Tensor<float>> input1 =
      std::make_shared<Tensor<float>>(3, 224, 224);
  input1->Fill(1.);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);

  std::vector<std::shared_ptr<Tensor<float>>> outputs =
      graph.Forward(inputs, false);
  ASSERT_EQ(outputs.size(), 1);

  const auto &output2 = CSVDataLoader::LoadData("../tmp/out.csv");
  const auto &output1 = outputs.front()->data().slice(0);
  ASSERT_EQ(output1.size(), output2.size());
  for (uint32_t s = 0; s < output1.size(); ++s) {
    ASSERT_LE(std::abs(output1.at(s) - output2.at(s)), 1e-4);
  }
}

void ResNetDemo(const std::vector<std::string> &image_paths,
                const std::string &param_path, const std::string &bin_path,
                const uint32_t batch_size) {
  using namespace kuiper_infer;
  std::map<std::string, double> run_duration_infos; //运行时间统计
  RuntimeGraph graph(param_path, bin_path);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  for (int i = 0; i < image_paths.size(); ++i) {
    std::string path = image_paths.at(i);
    cv::Mat image = cv::imread(path);
    // 图像预处理
    const auto &preprocess_start = std::chrono::steady_clock::now();
    sftensor input = PreProcessImage(image);
    const double preprocess_duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - preprocess_start)
            .count();
    LOG(INFO) << "Preprocess time cost: " << preprocess_duration << " s";
    std::vector<sftensor> inputs;
    inputs.push_back(input);

    // 推理100次，计算平均运行时间
    const auto &forward_start = std::chrono::steady_clock::now();
    u_int32_t run_times = 1;
    std::vector<sftensor> outputs;
    for (int j = 0; j < run_times; ++j) {
      outputs = graph.Forward(inputs, true);
    }
    const double forward_duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - forward_start)
            .count();
    LOG(INFO) << "Forward time cost: " << forward_duration / run_times << " s";

    // softmax
    const auto &softmax_start = std::chrono::steady_clock::now();
    std::vector<sftensor> outputs_softmax(batch_size);
    SoftmaxLayer softmax_layer;
    softmax_layer.Forward(outputs, outputs_softmax);
    assert(outputs_softmax.size() == batch_size);
    const double softmax_duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - softmax_start)
            .count();
    LOG(INFO) << "Softmax Layer time cost: " << softmax_duration << " s";

    for (int i = 0; i < outputs_softmax.size(); ++i) {
      const sftensor &output_tensor = outputs_softmax.at(i);
      assert(output_tensor->size() == 1 * 1000);
      // 找到类别概率最大的种类
      float max_prob = -1;
      int max_index = -1;
      for (int j = 0; j < output_tensor->size(); ++j) {
        float prob = output_tensor->index(j);
        if (max_prob <= prob) {
          max_prob = prob;
          max_index = j;
        }
      }
      printf("class with max prob is %f index %d\n", max_prob, max_index);
    }
  }
}

TEST(test_model, resnet_demo) {
  const uint32_t batch_size = 1;
  const std::vector<std::string> image_paths{"../tmp/dog.jpg"};
  const std::string &param_path = "../tmp/resnet18_batch1.pnnx.param";
  const std::string &bin_path = "../tmp/resnet18_batch1.pnnx.bin";

  ResNetDemo(image_paths, param_path, bin_path, batch_size);
}