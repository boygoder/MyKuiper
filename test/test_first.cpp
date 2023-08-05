//
// Created by fss on 22-12-13.
//

#include <armadillo>
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_first, demo1) {
  LOG(INFO) << "My first test!";
  arma::fmat in_1(32, 32, arma::fill::ones);
  ASSERT_EQ(in_1.n_cols, 32);
  ASSERT_EQ(in_1.n_rows, 32);
  ASSERT_EQ(in_1.size(), 32 * 32);
}

TEST(test_first, linear) {
  arma::fmat A = "1,2,3;"
                 "4,5,6;"
                 "7,8,9;";

  arma::fmat X = "1,1,1;"
                 "1,1,1;"
                 "1,1,1;";

  arma::fmat bias = "1,1,1;"
                    "1,1,1;"
                    "1,1,1;";

  arma::fmat output(3, 3);
  // todo 在此处插入代码，完成output = AxX + bias的运算
  output = A * X + bias;

  const uint32_t cols = 3;
  for (uint32_t c = 0; c < cols; ++c) {
    float *col_ptr = output.colptr(c);
    ASSERT_EQ(*(col_ptr + 0), 7);
    ASSERT_EQ(*(col_ptr + 1), 16);
    ASSERT_EQ(*(col_ptr + 2), 25);
  }
  LOG(INFO) << "\n"
            << "Result Passed!";
}

TEST(test_first, division) {
  arma::fcube A(3, 3, 1);
  arma::fcube B(3, 3, 1);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      A.at(i, j, 0) = 1.f;
      B.at(i, j, 0) = 2.f;
    }
  }

  arma::fcube output(3, 3, 1);
  // A/B
  output = A / B;

  const uint32_t cols = 3;
  for (uint32_t c = 0; c < cols; ++c) {
    float *col_ptr = output.slice(0).colptr(c);
    ASSERT_EQ(*(col_ptr + 0), 0.5f);
    ASSERT_EQ(*(col_ptr + 1), 0.5f);
    ASSERT_EQ(*(col_ptr + 2), 0.5f);
  }

  arma::fmat C(3, 3);
  arma::fmat D(3, 3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      C(i, j) = 1.f;
      D(i, j) = 2.f;
    }
  }
  arma::fmat result(3, 3);
  result = C / D;
  for (uint32_t c = 0; c < cols; ++c) {
    float *col_ptr = result.colptr(c);
    ASSERT_EQ(*(col_ptr + 0), 0.5);
    ASSERT_EQ(*(col_ptr + 1), 0.5);
    ASSERT_EQ(*(col_ptr + 2), 0.5);
  }
  LOG(INFO) << "\n"
            << "Result Passed!";
}
