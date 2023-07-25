/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/all_reduce_splitter.h"

#include <memory>

#include "absl/memory/memory.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_computation.h"
#include "xla/service/hlo_instruction.h"
#include "xla/service/hlo_matchers.h"
#include "xla/service/hlo_module.h"
#include "xla/service/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

using absl::nullopt;
using ::testing::AllOf;
namespace op = xla::testing::opcode_matchers;
int64_t kMaxCombineCount = 256;

int64_t AllReduceCount(const HloModule& module) {
  int64_t count = 0;
  for (HloComputation* computation : module.computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (HloInstruction* hlo : computation->instructions()) {
      if (hlo->opcode() == HloOpcode::kAllReduce) {
        ++count;
      }
    }
  }
  return count;
}

int64_t AllReduceOperandsCount(const HloModule& module) {
  int64_t count = 0;
  for (HloComputation* computation : module.computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (HloInstruction* hlo : computation->instructions()) {
      if (hlo->opcode() == HloOpcode::kAllReduce ||
          hlo->opcode() == HloOpcode::kAllReduceStart) {
        count += hlo->operand_count();
      }
    }
  }
  return count;
}

int64_t AllReduceSizeCount(const HloModule& module) {
  int64_t count = 0;
  for (HloComputation* computation : module.computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (HloInstruction* hlo : computation->instructions()) {
      if (hlo->opcode() == HloOpcode::kAllReduce ||
          hlo->opcode() == HloOpcode::kAllReduceStart) {
        for (HloInstruction* operands : hlo->operands()) {
          count += ShapeUtil::ByteSizeOf(operands->shape());
        }
      }
    }
  }
  return count;
}

int64_t MaxAllReduceOperandsCount(const HloModule& module) {
  int64_t max_count = 0;
  for (HloComputation* computation : module.computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (HloInstruction* hlo : computation->instructions()) {
      if (hlo->opcode() == HloOpcode::kAllReduceStart) {
        max_count = std::max(max_count, hlo->operand_count());
      }
    }
  }
  return max_count;
}

int64_t MaxAllReduceSizeCount(const HloModule& module) {
  int64_t max_size_count = 0;
  for (HloComputation* computation : module.computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (HloInstruction* hlo : computation->instructions()) {
      if (hlo->opcode() == HloOpcode::kAllReduceStart) {
        if (hlo->operand_count() < 2) continue;
        int64_t size_count = 0;
        for (HloInstruction* operands : hlo->operands()) {
          size_count += ShapeUtil::ByteSizeOf(operands->shape());
        }
        max_size_count = std::max(max_size_count, size_count);
      }
    }
  }
  return max_size_count;
}

// TODO(lyh271596): When I am more familiar with hlo, I need to use more focused
// hlo graphs to do unit tests, and for now I will use the results from the real
// model (VGG16) dump.
const char* const hlo_string = R"(
HloModule Module

AddComputation {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY entry {
  param0 = f32[1024, 1024] parameter(0)

  dot.0 = f32[1024, 1024] dot(%param0, %param0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot.1 = f32[1024, 1024] dot(%dot.0, %param0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot.2 = f32[1024, 1024] dot(%dot.1, %param0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot.3 = f32[1024, 1024] dot(%dot.2, %param0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot.4 = f32[1024, 1024] dot(%dot.3, %param0), lhs_contracting_dims={1}, rhs_contracting_dims={0}

  all-reduce = (f32[1024, 1024], f32[1024, 1024], f32[1024, 1024], f32[1024, 1024], f32[1024, 1024]) all-reduce(%dot.0, %dot.1, %dot.2, %dot.3, %dot.4), constrain_layout=true, replica_groups={}, to_apply=%AddComputation

  param1 = f32[1024, 1024] parameter(1)

  %get-tuple-element.0 = f32[1024, 1024] get-tuple-element(%all-reduce), index=0
  %get-tuple-element.1 = f32[1024, 1024] get-tuple-element(%all-reduce), index=1
  %get-tuple-element.2 = f32[1024, 1024] get-tuple-element(%all-reduce), index=2
  %get-tuple-element.3 = f32[1024, 1024] get-tuple-element(%all-reduce), index=3
  %get-tuple-element.4 = f32[1024, 1024] get-tuple-element(%all-reduce), index=4

  dot.5 = f32[1024, 1024] dot(%get-tuple-element.0, %param1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot.6 = f32[1024, 1024] dot(%get-tuple-element.1, %param1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot.7 = f32[1024, 1024] dot(%get-tuple-element.2, %param1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot.8 = f32[1024, 1024] dot(%get-tuple-element.3, %param1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot.9 = f32[1024, 1024] dot(%get-tuple-element.4, %param1), lhs_contracting_dims={1}, rhs_contracting_dims={0}

  ROOT tuple = (f32[1024, 1024], f32[1024, 1024], f32[1024, 1024], f32[1024, 1024], f32[1024, 1024]) tuple(%dot.5, %dot.6, %dot.7, %dot.8, %dot.9)
}

)";

using AllReduceSplitterTest = HloTestBase;

// Tests optimize of VGG 16 HLO graph.
TEST_F(AllReduceSplitterTest, Basic) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LOG(INFO) << "Original module:\n" << module->ToString();
  auto operand_count_before = AllReduceOperandsCount(*module);
  auto operand_size_before = AllReduceSizeCount(*module);
  AllReduceSplitter optimizer(
      /*combine_threshold_in_bytes=*/30 * 1024 * 1024,
      /*combine_threshold_count=*/256);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, optimizer.Run(module.get()));
  ASSERT_EQ(operand_count_before, AllReduceOperandsCount(*module));
  ASSERT_EQ(operand_size_before, AllReduceSizeCount(*module));
  ASSERT_TRUE(MaxAllReduceOperandsCount(*module) <= 256);
  ASSERT_TRUE(MaxAllReduceSizeCount(*module) <= 30 * 1024 * 1024);
  LOG(INFO) << "MaxAllReduceOperandsCount "
            << MaxAllReduceOperandsCount(*module);
  LOG(INFO) << "MaxAllReduceSizeCount " << MaxAllReduceSizeCount(*module);
  LOG(INFO) << operand_size_before << " vs " << AllReduceSizeCount(*module);
  LOG(INFO) << operand_count_before << " vs "
            << AllReduceOperandsCount(*module);
  LOG(INFO) << "After module:\n" << module->ToString();
}

TEST_F(AllReduceSplitterTest, Env) {
  int64_t combine_threshold_in_bytes = 0;
  int64_t combine_threshold_count = 0;
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto operand_count_before = AllReduceOperandsCount(*module);
  auto operand_size_before = AllReduceSizeCount(*module);
  tensorflow::setenv("XLA_GPU_ALL_REDUCE_COMBINE_THRESHOLD_IN_BYTES",
                     std::to_string(combine_threshold_in_bytes).c_str(), 1);
  tensorflow::setenv("XLA_GPU_ALL_REDUCE_COMBINE_THRESHOLD_COUNT",
                     std::to_string(combine_threshold_count).c_str(), 1);
  AllReduceSplitter optimizer(
      /*combine_threshold_in_bytes=*/30 * 1024 * 1024,
      /*combine_threshold_count=*/256);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, optimizer.Run(module.get()));
  ASSERT_EQ(operand_count_before, AllReduceOperandsCount(*module));
  ASSERT_EQ(operand_size_before, AllReduceSizeCount(*module));
}

TEST_F(AllReduceSplitterTest, EnvLoop) {
  for (size_t i = 1; i <= 1024; i *= 2) {
    for (size_t j = 1; j <= 1024; j *= 2) {
      int64_t combine_threshold_in_bytes = i * 1024 * 1024;
      int64_t combine_threshold_count = j;
      TF_ASSERT_OK_AND_ASSIGN(auto module,
                              ParseAndReturnVerifiedModule(hlo_string));
      tensorflow::setenv("XLA_GPU_ALL_REDUCE_COMBINE_THRESHOLD_IN_BYTES",
                         std::to_string(combine_threshold_in_bytes).c_str(), 1);
      tensorflow::setenv("XLA_GPU_ALL_REDUCE_COMBINE_THRESHOLD_COUNT",
                         std::to_string(combine_threshold_count).c_str(), 1);
      AllReduceSplitter optimizer(
          /*combine_threshold_in_bytes=*/30 * 1024 * 1024,
          /*combine_threshold_count=*/256);
      TF_ASSERT_OK_AND_ASSIGN(bool changed, optimizer.Run(module.get()));
      ASSERT_TRUE(MaxAllReduceSizeCount(*module) <= combine_threshold_in_bytes);
      ASSERT_TRUE(MaxAllReduceOperandsCount(*module) <=
                  combine_threshold_count);
    }
  }
}

}  // namespace
}  // namespace xla
