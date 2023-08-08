#include "../source/layer/details/maxpooling.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_layer, forward_maxpooling1) {
  using namespace kuiper_infer;
  uint32_t stride_h = 1;
  uint32_t stride_w = 1;
  uint32_t padding_h = 0;
  uint32_t padding_w = 0;
  uint32_t pooling_h = 2;
  uint32_t pooling_w = 2;

  std::shared_ptr<MaxPoolingLayer> max_layer =
      std::make_shared<MaxPoolingLayer>(padding_h, padding_w, pooling_h,
                                        pooling_w, stride_h, stride_w);
  CHECK(max_layer != nullptr);

  arma::fmat input_data = "0 1 2 ;"
                          "3 4 5 ;"
                          "6 7 8 ;";
  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(2, input_data.n_rows, input_data.n_cols);
  input->slice(0) = input_data;
  input->slice(1) = input_data;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  inputs.push_back(input);

  max_layer->Forward(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);
  const auto &output = outputs.at(0);
  LOG(INFO) << "\n" << output->data();
  ASSERT_EQ(output->rows(), 2);
  ASSERT_EQ(output->cols(), 2);

  ASSERT_EQ(output->at(0, 0, 0), 4);
  ASSERT_EQ(output->at(0, 0, 1), 5);
  ASSERT_EQ(output->at(0, 1, 0), 7);
  ASSERT_EQ(output->at(0, 1, 1), 8);

  ASSERT_EQ(output->at(1, 0, 0), 4);
  ASSERT_EQ(output->at(1, 0, 1), 5);
  ASSERT_EQ(output->at(1, 1, 0), 7);
  ASSERT_EQ(output->at(1, 1, 1), 8);
}