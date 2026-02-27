# ACLNN -> torch-npu coverage (from ACLNN list)

| ACLNN API | torch-npu | evidence | via (ATen ops) | direct match | suspected fusion | C++ funcs | remarks |
|---|---|---|---|---|---|---|---|
| aclnnAbs | 已接入 | yaml_exec | abs;abs.out;abs_ | abs;abs.out | False |  | shared_by_3_ops;yaml_only |
| aclnnAcos | 已接入 | yaml_exec | acos;acos.out | acos;acos.out | False |  | shared_by_2_ops;yaml_only |
| aclnnAcosh | 已接入 | yaml_exec | acosh;acosh.out | acosh;acosh.out | False |  | shared_by_2_ops;yaml_only |
| aclnnAdaLayerNorm | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnAdaLayerNormQuant | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnAdaLayerNormV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnAdaptiveAvgPool2d | 已接入 | src_scan | _adaptive_avg_pool2d;adaptive_avg_pool2d;adaptive_avg_pool2d.out | _adaptive_avg_pool2d;adaptive_avg_pool2d;adaptive_avg_pool2d.out | False | _adaptive_avg_pool2d;adaptive_avg_pool2d;adaptive_avg_pool2d_out | shared_by_3_ops;src_only |
| aclnnAdaptiveAvgPool2dBackward | 已接入 | yaml_exec | _adaptive_avg_pool2d_backward | _adaptive_avg_pool2d_backward | False |  | yaml_only |
| aclnnAdaptiveAvgPool3d | 已接入 | yaml_exec | _adaptive_avg_pool3d;adaptive_avg_pool3d.out | _adaptive_avg_pool3d;adaptive_avg_pool3d.out | False |  | shared_by_2_ops;yaml_only |
| aclnnAdaptiveAvgPool3dBackward | 已接入 | yaml_exec | _adaptive_avg_pool3d_backward;adaptive_avg_pool3d_backward.grad_input | _adaptive_avg_pool3d_backward;adaptive_avg_pool3d_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnAdaptiveMaxPool2d | 已接入 | yaml_exec | adaptive_max_pool2d;adaptive_max_pool2d.out | adaptive_max_pool2d;adaptive_max_pool2d.out | False |  | shared_by_2_ops;yaml_only |
| aclnnAdaptiveMaxPool2dBackward | 已接入 | yaml_exec | adaptive_max_pool2d_backward;adaptive_max_pool2d_backward.grad_input | adaptive_max_pool2d_backward;adaptive_max_pool2d_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnAdaptiveMaxPool3d | 已接入 | src_scan | adaptive_max_pool3d;adaptive_max_pool3d.out | adaptive_max_pool3d;adaptive_max_pool3d.out | False | adaptive_max_pool3d;adaptive_max_pool3d_out | shared_by_2_ops;src_only |
| aclnnAdaptiveMaxPool3dBackward | 已接入 | src_scan | adaptive_max_pool3d_backward;adaptive_max_pool3d_backward.grad_input | adaptive_max_pool3d_backward;adaptive_max_pool3d_backward.grad_input | False | adaptive_max_pool3d_backward;adaptive_max_pool3d_backward_out | shared_by_2_ops;src_only |
| aclnnAdd | 已接入 | src_scan | add.Scalar;add.Tensor;add.out | add.Scalar;add.Tensor;add.out | False | add;add_out;add_out_npu_nocheck | shared_by_3_ops;src_only |
| aclnnAddLayerNorm | 已接入 | src_scan | npu_add_layer_norm |  | True | npu_add_layer_norm | src_only |
| aclnnAddLayerNormGrad | 已接入 | src_scan | npu_add_layer_norm_backward |  | True | npu_add_layer_norm_backward | src_only |
| aclnnAddLora | 已接入 | src_scan | npu_batch_gather_matmul;npu_batch_gather_matmul_ |  | True | npu_batch_gather_matmul;npu_batch_gather_matmul_ | shared_by_2_ops;src_only |
| aclnnAddRelu | 已接入 | yaml_exec | _add_relu.Tensor;_add_relu.out | _add_relu.Tensor;_add_relu.out | False |  | shared_by_2_ops;yaml_only |
| aclnnAddRmsNorm | 已接入 | yaml_exec | npu_add_rms_norm |  | True |  | yaml_only |
| aclnnAddRmsNormCast | 已接入 | yaml_exec | npu_add_rms_norm_cast |  | True |  | yaml_only |
| aclnnAddRmsNormDynamicQuant | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnAddRmsNormDynamicQuantV2 | 已接入 | yaml_exec | npu_add_rms_norm_dynamic_quant |  | True |  | yaml_only |
| aclnnAddRmsNormQuant | 已接入 | src_scan | npu_add_rms_norm_quant |  | True | npu_add_rms_norm_quant | src_only |
| aclnnAddRmsNormQuantV2 | 已接入 | src_scan | npu_add_rms_norm_quant |  | True | npu_add_rms_norm_quant | src_only |
| aclnnAddbmm | 已接入 | src_scan | addbmm;addbmm.out;addbmm_ | addbmm;addbmm.out | False | addbmm;addbmm_;addbmm_out | shared_by_3_ops;src_only |
| aclnnAddcdiv | 已接入 | src_scan | addcdiv;addcdiv.out | addcdiv;addcdiv.out | False | addcdiv;addcdiv_out | shared_by_2_ops;src_only |
| aclnnAddcmul | 已接入 | src_scan | addcmul;addcmul.out | addcmul;addcmul.out | False | addcmul;addcmul_out | shared_by_2_ops;src_only |
| aclnnAddmm | 已接入 | src_scan | addmm;addmm.out;npu_linear | addmm;addmm.out | False | addmm;addmm_out;npu_linear | shared_by_3_ops;src_only |
| aclnnAddmmWeightNz | 已接入 | src_scan | addmm;addmm.out |  | True | addmm;addmm_out | shared_by_2_ops;src_only |
| aclnnAddmv | 已接入 | src_scan | addmv;addmv.out;addmv_ | addmv;addmv.out | False | addmv;addmv_;addmv_out;addmv_out_op_api | shared_by_3_ops;src_only |
| aclnnAddr | 已接入 | yaml_exec | addr;addr.out | addr;addr.out | False |  | shared_by_2_ops;yaml_only |
| aclnnAdds | 已接入 | src_scan | add.Scalar;add.Tensor;add.out |  | True | add;add_out;add_out_npu_nocheck | shared_by_3_ops;src_only |
| aclnnAdvanceStep | 已接入 | src_scan | npu_advance_step_flashattn |  | True | npu_advance_step_flashattn | src_only |
| aclnnAdvanceStepV2 | 已接入 | src_scan | npu_advance_step_flashattn |  | True | npu_advance_step_flashattn | src_only |
| aclnnAffineGrid | 已接入 | src_scan |  |  | True | OPS_ERROR | src_only;src_hit_but_op_name_unresolved |
| aclnnAll | 已接入 | src_scan | all;all.all_out;all.dim;all.out | all;all.all_out;all.dim;all.out | False | all;all_out | shared_by_4_ops;src_only |
| aclnnAllGatherMatmul | 已接入 | src_scan | npu_all_gather_base_mm |  | True | npu_all_gather_base_mm | src_only |
| aclnnAllGatherMatmulV2 | 已接入 | src_scan | npu_all_gather_base_mm;npu_all_gather_quant_mm |  | True | npu_all_gather_base_mm;npu_all_gather_quant_mm | shared_by_2_ops;src_only |
| aclnnAlltoAllAllGatherBatchMatMul | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnAlltoAllvGroupedMatMul | 已接入 | src_scan |  |  | True | options | src_only;src_hit_but_op_name_unresolved |
| aclnnAmax | 已接入 | yaml_exec | amax;amax.out | amax;amax.out | False |  | shared_by_2_ops;yaml_only |
| aclnnAmin | 已接入 | yaml_exec | amin;amin.out | amin;amin.out | False |  | shared_by_2_ops;yaml_only |
| aclnnAminmax | 已接入 | src_scan | aminmax.out | aminmax.out | False | aminmax_out | src_only |
| aclnnAminmaxAll | 已接入 | yaml_exec | _aminmax |  | True |  | yaml_only |
| aclnnAminmaxDim | 已接入 | yaml_exec | _aminmax.dim |  | True |  | yaml_only |
| aclnnAny | 已接入 | src_scan | any;any.all_out;any.dim;any.out | any;any.all_out;any.dim;any.out | False | any;any_out | shared_by_4_ops;src_only |
| aclnnApplyAdamW | 已接入 | src_scan | npu_apply_adam_w.out |  | True | npu_apply_adam_w_out | src_only |
| aclnnApplyAdamWQuant | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnApplyAdamWV2 | 已接入 | src_scan |  |  | True | size | src_only;src_hit_but_op_name_unresolved |
| aclnnApplyFusedEmaAdam | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnApplyRotaryPosEmb | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnApplyRotaryPosEmbV2 | 已接入 | src_scan | npu_apply_rotary_pos_emb |  | True | npu_apply_rotary_pos_emb | src_only |
| aclnnApplyTopKTopP | 已接入 | yaml_exec | npu_top_k_top_p |  | True |  | yaml_only |
| aclnnArange | 已接入 | src_scan | arange;arange.out;arange.start;arange.start_out;arange.start_step | arange;arange.out;arange.start;arange.start_out;arange.start_step | False | arange;arange_out;arange_out_op_api | shared_by_5_ops;src_only |
| aclnnArgMax | 已接入 | src_scan | argmax.out |  | True | argmax_exec;argmax_out | src_only |
| aclnnArgMin | 已接入 | src_scan | argmin;argmin.out |  | True | argmin;argmin_exec;argmin_out | shared_by_2_ops;src_only |
| aclnnArgsort | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnAscendAntiQuant | 已接入 | src_scan | npu_anti_quant |  | True | npu_anti_quant | src_only |
| aclnnAscendQuant | 已接入 | src_scan |  |  | True | npu_quantize_by_ascend_quant | src_only;src_hit_but_op_name_unresolved |
| aclnnAscendQuantV3 | 已接入 | src_scan |  |  | True | npu_quantize_by_ascend_quant | src_only;src_hit_but_op_name_unresolved |
| aclnnAsin | 已接入 | yaml_exec | asin;asin.out | asin;asin.out | False |  | shared_by_2_ops;yaml_only |
| aclnnAsinh | 已接入 | yaml_exec | asinh;asinh.out | asinh;asinh.out | False |  | shared_by_2_ops;yaml_only |
| aclnnAtan | 已接入 | yaml_exec | atan;atan.out | atan;atan.out | False |  | shared_by_2_ops;yaml_only |
| aclnnAtan2 | 已接入 | src_scan | atan2;atan2.out | atan2;atan2.out | False | atan2;atan2_out | shared_by_2_ops;src_only |
| aclnnAtanh | 已接入 | yaml_exec | atanh;atanh.out | atanh;atanh.out | False |  | shared_by_2_ops;yaml_only |
| aclnnAttentionUpdate | 已接入 | src_scan | npu_attention_update |  | True | npu_attention_update | src_only |
| aclnnAvgPool2d | 已接入 | src_scan | avg_pool2d;avg_pool2d.out | avg_pool2d;avg_pool2d.out | False | avg_pool2d;avg_pool2d_out;avg_pool2d_out_npu_nocheck_opapi | shared_by_2_ops;src_only |
| aclnnAvgPool2dBackward | 已接入 | src_scan | avg_pool2d_backward;avg_pool2d_backward.grad_input | avg_pool2d_backward;avg_pool2d_backward.grad_input | False | avg_pool2d_backward;avg_pool2d_backward_out;avg_pool2d_backward_out_npu_nocheck_api | shared_by_2_ops;src_only |
| aclnnAvgPool3d | 已接入 | src_scan | avg_pool3d;avg_pool3d.out | avg_pool3d;avg_pool3d.out | False | avg_pool3d;avg_pool3d_out;avg_pool3d_out_npu_nocheck_opapi | shared_by_2_ops;src_only |
| aclnnAvgPool3dBackward | 已接入 | src_scan | avg_pool3d_backward;avg_pool3d_backward.grad_input | avg_pool3d_backward;avg_pool3d_backward.grad_input | False | avg_pool3d_backward;avg_pool3d_backward_out;avg_pool3d_backward_out_npu_nocheck_api | shared_by_2_ops;src_only |
| aclnnBackgroundReplace | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnBaddbmm | 已接入 | src_scan | baddbmm;baddbmm.out;baddbmm_ | baddbmm;baddbmm.out | False | baddbmm;baddbmm_;baddbmm_out | shared_by_3_ops;src_only |
| aclnnBatchMatMul | 已接入 | src_scan | affine_grid_generator_backward;bmm;bmm.out |  | True | affine_grid_generator_backward;bmm;bmm_out | shared_by_3_ops;src_only |
| aclnnBatchMatMulReduceScatterAlltoAll | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnBatchMatMulWeightNz | 已接入 | src_scan | bmm;bmm.out |  | True | bmm;bmm_out | shared_by_2_ops;src_only |
| aclnnBatchMatmulQuant | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnBatchNorm | 已接入 | yaml_exec | native_batch_norm;native_batch_norm.out |  | True |  | shared_by_2_ops;yaml_only |
| aclnnBatchNormBackward | 已接入 | src_scan | native_batch_norm_backward |  | True | native_batch_norm_backward | src_only |
| aclnnBatchNormElemt | 已接入 | src_scan | batch_norm_elemt;batch_norm_elemt.out | batch_norm_elemt;batch_norm_elemt.out | False | batch_norm_elemt;batch_norm_elemt_out | shared_by_2_ops;src_only |
| aclnnBatchNormElemtBackward | 已接入 | yaml_exec | batch_norm_backward_elemt |  | True |  | yaml_only |
| aclnnBatchNormGatherStatsWithCounts | 已接入 | yaml_exec | batch_norm_gather_stats_with_counts | batch_norm_gather_stats_with_counts | False |  | yaml_only |
| aclnnBatchNormReduce | 已接入 | src_scan | batch_norm_reduce | batch_norm_reduce | False | batch_norm_reduce | src_only |
| aclnnBatchNormReduceBackward | 已接入 | src_scan | batch_norm_backward_reduce |  | True | batch_norm_backward_reduce | src_only |
| aclnnBatchNormStats | 已接入 | yaml_exec | batch_norm_stats | batch_norm_stats | False |  | yaml_only |
| aclnnBernoulli | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnBernoulliTensor | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnBidirectionLSTM | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnBidirectionLSTMV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnBinaryCrossEntropy | 已接入 | yaml_exec | binary_cross_entropy;binary_cross_entropy.out | binary_cross_entropy;binary_cross_entropy.out | False |  | shared_by_2_ops;yaml_only |
| aclnnBinaryCrossEntropyBackward | 已接入 | yaml_exec | binary_cross_entropy_backward;binary_cross_entropy_backward.grad_input | binary_cross_entropy_backward;binary_cross_entropy_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnBinaryCrossEntropyWithLogits | 已接入 | yaml_exec | binary_cross_entropy_with_logits | binary_cross_entropy_with_logits | False |  | yaml_only |
| aclnnBinaryCrossEntropyWithLogitsBackward | 已接入 | yaml_exec | npu_binary_cross_entropy_with_logits_backward |  | True |  | yaml_only |
| aclnnBinaryCrossEntropyWithLogitsTargetBackward | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnBincount | 已接入 | src_scan | bincount | bincount | False | bincount | src_only |
| aclnnBitwiseAndScalar | 已接入 | src_scan | bitwise_and.Scalar;bitwise_and.Scalar_out;bitwise_and.Tensor;bitwise_and.Tensor_out |  | True | bitwise_and;bitwise_and_op_api_out_npu_nocheck;bitwise_and_out | shared_by_4_ops;src_only |
| aclnnBitwiseAndTensor | 已接入 | src_scan | bitwise_and.Scalar;bitwise_and.Scalar_out;bitwise_and.Tensor;bitwise_and.Tensor_out |  | True | bitwise_and;bitwise_and_op_api_out_npu_nocheck;bitwise_and_out | shared_by_4_ops;src_only |
| aclnnBitwiseNot | 已接入 | yaml_exec | bitwise_not;bitwise_not.out;bitwise_not_ | bitwise_not;bitwise_not.out | False |  | shared_by_3_ops;yaml_only |
| aclnnBitwiseOrScalar | 已接入 | src_scan | bitwise_or.Scalar;bitwise_or.Scalar_out;bitwise_or.Tensor;bitwise_or.Tensor_out |  | True | bitwise_or;bitwise_or_out;bitwise_or_out_nocheck | shared_by_4_ops;src_only |
| aclnnBitwiseOrTensor | 已接入 | src_scan | bitwise_or.Scalar;bitwise_or.Scalar_out;bitwise_or.Tensor;bitwise_or.Tensor_out |  | True | bitwise_or;bitwise_or_out;bitwise_or_out_nocheck | shared_by_4_ops;src_only |
| aclnnBitwiseXorScalar | 已接入 | src_scan | bitwise_xor.Scalar;bitwise_xor.Scalar_out;bitwise_xor.Tensor;bitwise_xor.Tensor_out |  | True | bitwise_xor;bitwise_xor_out;bitwise_xor_out_nocheck | shared_by_4_ops;src_only |
| aclnnBitwiseXorTensor | 已接入 | src_scan | bitwise_xor.Scalar;bitwise_xor.Scalar_out;bitwise_xor.Tensor;bitwise_xor.Tensor_out |  | True | bitwise_xor;bitwise_xor_out;bitwise_xor_out_nocheck | shared_by_4_ops;src_only |
| aclnnBlendImagesCustom | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnCalculateConvolutionWeightSize | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnCalculateMatmulWeightSize | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnCalculateMatmulWeightSizeV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnCast | 已接入 | src_scan | _npu_dtype_cast;npu_dtype_cast |  | True | _npu_dtype_cast;npu_dtype_cast;npu_dtype_cast_impl_op_api | shared_by_2_ops;src_only |
| aclnnCat | 已接入 | src_scan | cat;cat.names;cat.names_out;cat.out | cat;cat.names;cat.names_out;cat.out | False | cat;cat_out | shared_by_4_ops;src_only |
| aclnnCeil | 已接入 | yaml_exec | ceil;ceil.out | ceil;ceil.out | False |  | shared_by_2_ops;yaml_only |
| aclnnCelu | 已接入 | yaml_exec | celu | celu | False |  | yaml_only |
| aclnnChamferDistanceBackward | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnChannelShuffle | 已接入 | yaml_exec | channel_shuffle | channel_shuffle | False |  | yaml_only |
| aclnnCircularPad2d | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnCircularPad2dBackward | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnCircularPad3d | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnCircularPad3dBackward | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnClamp | 已接入 | yaml_exec | clamp;clamp.out;clamp_ | clamp;clamp.out | False |  | shared_by_3_ops;yaml_only |
| aclnnClampMax | 已接入 | yaml_exec | clamp_max;clamp_max.out | clamp_max;clamp_max.out | False |  | shared_by_2_ops;yaml_only |
| aclnnClampMaxTensor | 已接入 | yaml_exec | clamp_max.Tensor;clamp_max.Tensor_out |  | True |  | shared_by_2_ops;yaml_only |
| aclnnClampMin | 已接入 | yaml_exec | clamp_min;clamp_min.out;clamp_min_ | clamp_min;clamp_min.out | False |  | shared_by_3_ops;yaml_only |
| aclnnClampMinTensor | 已接入 | yaml_exec | clamp_min.Tensor;clamp_min.Tensor_out |  | True |  | shared_by_2_ops;yaml_only |
| aclnnClampTensor | 已接入 | yaml_exec | clamp.Tensor;clamp.Tensor_out;clamp_.Tensor |  | True |  | shared_by_3_ops;yaml_only |
| aclnnClippedSwiglu | 已接入 | yaml_exec | npu_clipped_swiglu |  | True |  | yaml_only |
| aclnnComplex | 已接入 | src_scan | complex;complex.out | complex;complex.out | False | complex;complex_out | shared_by_2_ops;src_only |
| aclnnConstantPadNd | 已接入 | src_scan | constant_pad_nd | constant_pad_nd | False | constant_pad_nd | src_only |
| aclnnConvDepthwise2d | 已接入 | src_scan | _conv_depthwise2d;_conv_depthwise2d.out | _conv_depthwise2d;_conv_depthwise2d.out | False | _conv_depthwise2d;_conv_depthwise2d_out | shared_by_2_ops;src_only |
| aclnnConvTbc | 已接入 | yaml_exec | conv_tbc | conv_tbc | False |  | yaml_only |
| aclnnConvTbcBackward | 已接入 | src_scan | conv_tbc_backward | conv_tbc_backward | False | conv_tbc_backward | src_only |
| aclnnConvertWeightToINT4Pack | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnConvolution | 已接入 | src_scan | _convolution;_nnpack_spatial_convolution;_slow_conv2d_forward;_slow_conv2d_forward.output;convolution_overrideable;slow_conv3d_forward;slow_conv3d_forward.output;slow_conv_dilated2d;slow_conv_transpose2d;slow_conv_transpose2d.out | _convolution | False | _calc_convolution;_convolution;_nnpack_spatial_convolution;_slow_conv2d_forward;_slow_conv2d_forward_out;convolution_overrideable;slow_conv3d_forward;slow_conv3d_forward_out;...(+3) | shared_by_10_ops;src_only |
| aclnnConvolutionBackward | 已接入 | src_scan | _slow_conv2d_backward.output_mask;convolution_backward;convolution_backward_overrideable;slow_conv_dilated2d_backward;slow_conv_transpose2d_backward | convolution_backward | False | _calc_convolution_backward;_slow_conv2d_backward;convolution_backward;convolution_backward_overrideable;slow_conv_dilated2d_backward;slow_conv_transpose2d_backward | shared_by_5_ops;src_only |
| aclnnCos | 已接入 | yaml_exec | cos;cos.out | cos;cos.out | False |  | shared_by_2_ops;yaml_only |
| aclnnCosh | 已接入 | yaml_exec | cosh;cosh.out | cosh;cosh.out | False |  | shared_by_2_ops;yaml_only |
| aclnnCrossEntropyLoss | 已接入 | yaml_exec | npu_cross_entropy_loss |  | True |  | yaml_only |
| aclnnCrossEntropyLossGrad | 已接入 | yaml_exec | npu_cross_entropy_loss_backward |  | True |  | yaml_only |
| aclnnCtcLoss | 已接入 | src_scan | _ctc_loss | _ctc_loss | False | _ctc_loss | src_only |
| aclnnCtcLossBackward | 已接入 | yaml_exec | _ctc_loss_backward | _ctc_loss_backward | False |  | yaml_only |
| aclnnCummax | 已接入 | src_scan | _cummax_helper |  | True | _cummax_helper | src_only |
| aclnnCummin | 已接入 | src_scan | _cummin_helper |  | True | _cummin_helper | src_only |
| aclnnCumprod | 已接入 | yaml_exec | cumprod.out | cumprod.out | False |  | yaml_only |
| aclnnCumsum | 已接入 | src_scan | cumsum;cumsum.dimname_out;cumsum.out | cumsum;cumsum.dimname_out;cumsum.out | False | cumsum;cumsum_out | shared_by_3_ops;src_only |
| aclnnCumsumV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnDeepNorm | 已接入 | src_scan | npu_deep_norm |  | True | npu_deep_norm | src_only |
| aclnnDeepNormGrad | 已接入 | src_scan | npu_deep_norm_backward |  | True | npu_deep_norm_backward | src_only |
| aclnnDeformableConv2d | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnDequantBias | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnDequantRopeQuantKvcache | 已接入 | src_scan | npu_dequant_rope_quant_kvcache;npu_rope_quant_kvcache |  | True | npu_dequant_rope_quant_kvcache;npu_rope_quant_kvcache | shared_by_2_ops;src_only |
| aclnnDequantSwigluQuant | 已接入 | src_scan | npu_dequant_swiglu_quant |  | True | npu_dequant_swiglu_quant | src_only |
| aclnnDiag | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnDiagFlat | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnDigamma | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnDistributeBarrier | 已接入 | src_scan | _npu_distribute_barrier |  | True | _npu_distribute_barrier | src_only |
| aclnnDistributeBarrierV2 | 已接入 | src_scan | _npu_distribute_barrier |  | True | _npu_distribute_barrier | src_only |
| aclnnDiv | 已接入 | src_scan | div.Scalar;div.Scalar_mode;div.Tensor;div.Tensor_mode;div.out;div.out_mode | div.Scalar;div.Scalar_mode;div.Tensor;div.Tensor_mode;div.out;div.out_mode | False | div;div_out;div_out_npu_opapi_nocheck | shared_by_6_ops;src_only |
| aclnnDivMod | 已接入 | src_scan | div.Scalar;div.Scalar_mode;div.Tensor;div.Tensor_mode;div.out;div.out_mode |  | True | div;div_out | shared_by_6_ops;src_only |
| aclnnDivMods | 已接入 | src_scan | div.Scalar;div.Scalar_mode;div.Tensor;div.Tensor_mode;div.out;div.out_mode |  | True | div;div_out | shared_by_6_ops;src_only |
| aclnnDivs | 已接入 | src_scan | div.Scalar;div.Scalar_mode;div.Tensor;div.Tensor_mode;div.out;div.out_mode |  | True | div;div_out;div_out_npu_opapi_nocheck | shared_by_6_ops;src_only |
| aclnnDot | 已接入 | yaml_exec | dot;dot.out;vdot;vdot.out | dot;dot.out | False |  | shared_by_4_ops;yaml_only |
| aclnnDropout | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnDropoutBackward | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnDropoutDoMask | 已接入 | src_scan | _npu_dropout;native_dropout;native_dropout_backward;npu_dropout_backward |  | True | _npu_dropout;native_dropout;native_dropout_backward;npu_dropout_backward | shared_by_4_ops;src_only |
| aclnnDropoutGenMask | 已接入 | src_scan | npu_dropout_gen_mask |  | True | npu_dropout_gen_mask | src_only |
| aclnnDropoutGenMaskV2 | 已接入 | src_scan | _npu_dropout;_npu_dropout_gen_mask.Tensor;native_dropout |  | True | _npu_dropout;_npu_dropout_gen_mask;dropout_gen_mask_impl;dropout_gen_mask_tensor_impl;gen_mask_impl;native_dropout | shared_by_3_ops;src_only |
| aclnnDropoutGenMaskV2Tensor | 已接入 | src_scan | _npu_dropout |  | True | _npu_dropout;dropout_gen_mask_tensor_impl | src_only |
| aclnnDynamicBlockQuant | 已接入 | src_scan | npu_dynamic_block_quant |  | True | npu_dynamic_block_quant | src_only |
| aclnnDynamicQuant | 已接入 | src_scan |  |  | True | npu_dynamic_quant_v0 | src_only;src_hit_but_op_name_unresolved |
| aclnnDynamicQuantV2 | 已接入 | src_scan | npu_dynamic_quant;npu_dynamic_quant_asymmetric |  | True | npu_dynamic_quant;npu_dynamic_quant_asymmetric | shared_by_2_ops;src_only |
| aclnnEinsum | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnElu | 已接入 | yaml_exec | elu;elu.out | elu;elu.out | False |  | shared_by_2_ops;yaml_only |
| aclnnEluBackward | 已接入 | yaml_exec | elu_backward;elu_backward.grad_input | elu_backward;elu_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnEmbedding | 已接入 | src_scan |  |  | True | embedding_symint | src_only;src_hit_but_op_name_unresolved |
| aclnnEmbeddingBag | 已接入 | src_scan | _embedding_bag;_embedding_bag_forward_only | _embedding_bag | False | _embedding_bag;_embedding_bag_forward_only | shared_by_2_ops;src_only |
| aclnnEmbeddingDenseBackward | 已接入 | yaml_exec | embedding_dense_backward | embedding_dense_backward | False |  | yaml_only |
| aclnnEmbeddingRenorm | 已接入 | src_scan | embedding_renorm_ |  | True | embedding_renorm_ | src_only |
| aclnnEqScalar | 已接入 | src_scan | eq.Scalar;eq.Scalar_out;eq.Tensor;eq.Tensor_out |  | True | eq;eq_out;eq_out_npu_scalar | shared_by_4_ops;src_only |
| aclnnEqTensor | 已接入 | src_scan | eq.Scalar;eq.Scalar_out;eq.Tensor;eq.Tensor_out |  | True | eq;eq_out | shared_by_4_ops;src_only |
| aclnnEqual | 已接入 | src_scan | equal | equal | False | equal | src_only |
| aclnnErf | 已接入 | yaml_exec | erf;erf.out | erf;erf.out | False |  | shared_by_2_ops;yaml_only |
| aclnnErfc | 已接入 | yaml_exec | erfc;erfc.out | erfc;erfc.out | False |  | shared_by_2_ops;yaml_only |
| aclnnErfinv | 已接入 | yaml_exec | erfinv;erfinv.out | erfinv;erfinv.out | False |  | shared_by_2_ops;yaml_only |
| aclnnExp | 已接入 | yaml_exec | exp;exp.out | exp;exp.out | False |  | shared_by_2_ops;yaml_only |
| aclnnExp2 | 已接入 | yaml_exec | exp2;exp2.out | exp2;exp2.out | False |  | shared_by_2_ops;yaml_only |
| aclnnExpSegsum | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnExpSegsumBackward | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnExpand | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnExpandIntoJaggedPermute | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnExpm1 | 已接入 | yaml_exec | expm1;expm1.out | expm1;expm1.out | False |  | shared_by_2_ops;yaml_only |
| aclnnEye | 已接入 | src_scan | eye;eye.m;eye.m_out;eye.out | eye;eye.m;eye.m_out;eye.out | False | eye;eye_out | shared_by_4_ops;src_only |
| aclnnFFN | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFFNV2 | 已接入 | src_scan | npu_ffn |  | True | npu_ffn | src_only |
| aclnnFFNV3 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFakeQuantPerChannelAffineCachemask | 已接入 | src_scan | fake_quantize_per_channel_affine_cachemask |  | True | fake_quantize_per_channel_affine_cachemask | src_only |
| aclnnFakeQuantPerTensorAffineCachemask | 已接入 | src_scan | _fake_quantize_per_tensor_affine_cachemask_tensor_qparams |  | True | _fake_quantize_per_tensor_affine_cachemask_tensor_qparams | src_only |
| aclnnFastBatchNormBackward | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFastGelu | 已接入 | src_scan | npu_fast_gelu |  | True | npu_fast_gelu | src_only |
| aclnnFastGeluBackward | 已接入 | src_scan | npu_fast_gelu_backward |  | True | npu_fast_gelu_backward | src_only |
| aclnnFatreluMul | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFinalize | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlashAttentionScore | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlashAttentionScoreGrad | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlashAttentionScoreGradV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlashAttentionScoreGradV3 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlashAttentionScoreV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlashAttentionScoreV3 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlashAttentionUnpaddingScoreGrad | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlashAttentionUnpaddingScoreGradV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlashAttentionUnpaddingScoreGradV3 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlashAttentionUnpaddingScoreGradV4 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlashAttentionUnpaddingScoreGradV5 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlashAttentionVarLenScore | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlashAttentionVarLenScoreV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlashAttentionVarLenScoreV3 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlashAttentionVarLenScoreV4 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlashAttentionVarLenScoreV5 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlatQuant | 已接入 | src_scan | npu_kronecker_quant |  | True | npu_kronecker_quant | src_only |
| aclnnFlatten | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFlip | 已接入 | yaml_exec | flip | flip | False |  | yaml_only |
| aclnnFloor | 已接入 | yaml_exec | floor;floor.out | floor;floor.out | False |  | shared_by_2_ops;yaml_only |
| aclnnFloorDivide | 已接入 | src_scan | floor_divide;floor_divide.Scalar;floor_divide.out | floor_divide;floor_divide.Scalar;floor_divide.out | False | floor_divide;floor_divide_out;floor_divide_out_npu_opapi | shared_by_3_ops;src_only |
| aclnnFloorDivides | 已接入 | src_scan | floor_divide;floor_divide.Scalar;floor_divide.out |  | True | floor_divide;floor_divide_out;floor_divide_out_npu_opapi | shared_by_3_ops;src_only |
| aclnnFmodScalar | 已接入 | yaml_exec | fmod.Scalar;fmod.Scalar_out |  | True |  | shared_by_2_ops;yaml_only |
| aclnnFmodTensor | 已接入 | yaml_exec | fmod.Tensor;fmod.Tensor_out |  | True |  | shared_by_2_ops;yaml_only |
| aclnnForeachAbs | 已接入 | src_scan | _foreach_abs;_foreach_abs_ | _foreach_abs | False | _foreach_abs;_foreach_abs_;_split_and_exec_npu_cmd_abs | shared_by_2_ops;src_only |
| aclnnForeachAcos | 已接入 | src_scan | _foreach_acos;_foreach_acos_ | _foreach_acos | False | _foreach_acos;_foreach_acos_;_split_and_exec_npu_cmd_acos | shared_by_2_ops;src_only |
| aclnnForeachAddList | 已接入 | src_scan |  |  | True | _foreach_add_v1;_foreach_add_v1_ | src_only;src_hit_but_op_name_unresolved |
| aclnnForeachAddListV2 | 已接入 | src_scan | _foreach_add.List;_foreach_add.Scalar;_foreach_add.ScalarList;_foreach_add_.List;_foreach_add_.Scalar;_foreach_add_.ScalarList |  | True | _foreach_add;_foreach_add_;_split_and_exec_npu_cmd_add | shared_by_6_ops;src_only |
| aclnnForeachAddScalar | 已接入 | src_scan |  |  | True | _foreach_add_v1;_foreach_add_v1_ | src_only;src_hit_but_op_name_unresolved |
| aclnnForeachAddScalarList | 已接入 | src_scan | _foreach_add.List;_foreach_add.Scalar;_foreach_add.ScalarList;_foreach_add_.List;_foreach_add_.Scalar;_foreach_add_.ScalarList |  | True | _foreach_add;_foreach_add_;_split_and_exec_npu_cmd_add_scalarlist | shared_by_6_ops;src_only |
| aclnnForeachAddScalarV2 | 已接入 | src_scan | _foreach_add.List;_foreach_add.Scalar;_foreach_add.ScalarList;_foreach_add_.List;_foreach_add_.Scalar;_foreach_add_.ScalarList |  | True | _foreach_add;_foreach_add_;_split_and_exec_npu_cmd_add_scalar | shared_by_6_ops;src_only |
| aclnnForeachAddcdivList | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnForeachAddcdivScalar | 已接入 | src_scan |  |  | True | _foreach_addcdiv_v1;_foreach_addcdiv_v1_ | src_only;src_hit_but_op_name_unresolved |
| aclnnForeachAddcdivScalarList | 已接入 | src_scan |  |  | True | _split_and_exec_npu_cmd_addcdiv_tensor | src_only;src_hit_but_op_name_unresolved |
| aclnnForeachAddcdivScalarV2 | 已接入 | src_scan | _foreach_addcdiv.Scalar;_foreach_addcdiv.ScalarList;_foreach_addcdiv.Tensor;_foreach_addcdiv_.Scalar;_foreach_addcdiv_.ScalarList;_foreach_addcdiv_.Tensor |  | True | _foreach_addcdiv;_foreach_addcdiv_;_split_and_exec_npu_cmd_addcdiv_scalar | shared_by_6_ops;src_only |
| aclnnForeachAddcmulList | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnForeachAddcmulScalar | 已接入 | src_scan |  |  | True | _foreach_addcmul_v1;_foreach_addcmul_v1_ | src_only;src_hit_but_op_name_unresolved |
| aclnnForeachAddcmulScalarList | 已接入 | src_scan |  |  | True | _split_and_exec_npu_cmd_addcmul_tensor | src_only;src_hit_but_op_name_unresolved |
| aclnnForeachAddcmulScalarV2 | 已接入 | src_scan | _foreach_addcmul.Scalar;_foreach_addcmul.ScalarList;_foreach_addcmul.Tensor;_foreach_addcmul_.Scalar;_foreach_addcmul_.ScalarList;_foreach_addcmul_.Tensor |  | True | _foreach_addcmul;_foreach_addcmul_;_split_and_exec_npu_cmd_addcmul_scalar | shared_by_6_ops;src_only |
| aclnnForeachAsin | 已接入 | src_scan | _foreach_asin;_foreach_asin_ | _foreach_asin | False | _foreach_asin;_foreach_asin_;_split_and_exec_npu_cmd_asin | shared_by_2_ops;src_only |
| aclnnForeachAtan | 已接入 | src_scan | _foreach_atan;_foreach_atan_ | _foreach_atan | False | _foreach_atan;_foreach_atan_;_split_and_exec_npu_cmd_atan | shared_by_2_ops;src_only |
| aclnnForeachCopy | 已接入 | src_scan | _foreach_copy_ |  | True | _foreach_copy_;exec_npu_cmd_copy | src_only |
| aclnnForeachCos | 已接入 | src_scan | _foreach_cos;_foreach_cos_ | _foreach_cos | False | _foreach_cos;_foreach_cos_;_split_and_exec_npu_cmd_cos | shared_by_2_ops;src_only |
| aclnnForeachCosh | 已接入 | src_scan | _foreach_cosh;_foreach_cosh_ | _foreach_cosh | False | _foreach_cosh;_foreach_cosh_;_split_and_exec_npu_cmd_cosh | shared_by_2_ops;src_only |
| aclnnForeachDivList | 已接入 | src_scan | _foreach_div.List;_foreach_div.Scalar;_foreach_div.ScalarList;_foreach_div_.List;_foreach_div_.Scalar;_foreach_div_.ScalarList |  | True | _foreach_div;_foreach_div_;_split_and_exec_npu_cmd_div | shared_by_6_ops;src_only |
| aclnnForeachDivScalar | 已接入 | src_scan |  |  | True | _foreach_div_v1;_foreach_div_v1_ | src_only;src_hit_but_op_name_unresolved |
| aclnnForeachDivScalarList | 已接入 | src_scan | _foreach_div.List;_foreach_div.Scalar;_foreach_div.ScalarList;_foreach_div_.List;_foreach_div_.Scalar;_foreach_div_.ScalarList |  | True | _foreach_div;_foreach_div_;_split_and_exec_npu_cmd_div_scalar_list | shared_by_6_ops;src_only |
| aclnnForeachDivScalarV2 | 已接入 | src_scan | _foreach_div.List;_foreach_div.Scalar;_foreach_div.ScalarList;_foreach_div_.List;_foreach_div_.Scalar;_foreach_div_.ScalarList |  | True | _foreach_div;_foreach_div_;_split_and_exec_npu_cmd_div_scalar | shared_by_6_ops;src_only |
| aclnnForeachErf | 已接入 | src_scan | _foreach_erf;_foreach_erf_ | _foreach_erf | False | _foreach_erf;_foreach_erf_;_split_and_exec_npu_cmd_erf | shared_by_2_ops;src_only |
| aclnnForeachErfc | 已接入 | src_scan | _foreach_erfc;_foreach_erfc_ | _foreach_erfc | False | _foreach_erfc;_foreach_erfc_;_split_and_exec_npu_cmd_erfc | shared_by_2_ops;src_only |
| aclnnForeachExp | 已接入 | src_scan | _foreach_exp;_foreach_exp_ | _foreach_exp | False | _foreach_exp;_foreach_exp_;_split_and_exec_npu_cmd_exp | shared_by_2_ops;src_only |
| aclnnForeachExpm1 | 已接入 | src_scan | _foreach_expm1;_foreach_expm1_ | _foreach_expm1 | False | _foreach_expm1;_foreach_expm1_;_split_and_exec_npu_cmd_expm1 | shared_by_2_ops;src_only |
| aclnnForeachLerpList | 已接入 | src_scan | _foreach_lerp.List;_foreach_lerp.Scalar;_foreach_lerp_.List;_foreach_lerp_.Scalar |  | True | _foreach_lerp;_foreach_lerp_ | shared_by_4_ops;src_only |
| aclnnForeachLerpScalar | 已接入 | src_scan | _foreach_lerp.List;_foreach_lerp.Scalar;_foreach_lerp_.List;_foreach_lerp_.Scalar |  | True | _foreach_lerp;_foreach_lerp_;adaptToDouble | shared_by_4_ops;src_only |
| aclnnForeachLog | 已接入 | src_scan | _foreach_log;_foreach_log_ | _foreach_log | False | _foreach_log;_foreach_log_;_split_and_exec_npu_cmd_log | shared_by_2_ops;src_only |
| aclnnForeachLog10 | 已接入 | src_scan | _foreach_log10;_foreach_log10_ | _foreach_log10 | False | _foreach_log10;_foreach_log10_;_split_and_exec_npu_cmd_log10 | shared_by_2_ops;src_only |
| aclnnForeachLog1p | 已接入 | src_scan | _foreach_log1p;_foreach_log1p_ | _foreach_log1p | False | _foreach_log1p;_foreach_log1p_;_split_and_exec_npu_cmd_log1p | shared_by_2_ops;src_only |
| aclnnForeachLog2 | 已接入 | src_scan | _foreach_log2;_foreach_log2_ | _foreach_log2 | False | _foreach_log2;_foreach_log2_;_split_and_exec_npu_cmd_log2 | shared_by_2_ops;src_only |
| aclnnForeachMaximumList | 已接入 | src_scan | _foreach_maximum.List;_foreach_maximum.Scalar;_foreach_maximum.ScalarList;_foreach_maximum_.List;_foreach_maximum_.Scalar;_foreach_maximum_.ScalarList |  | True | _foreach_maximum;_foreach_maximum_;_split_and_exec_npu_cmd_max | shared_by_6_ops;src_only |
| aclnnForeachMaximumScalar | 已接入 | src_scan |  |  | True | _foreach_maximum_v1;_foreach_maximum_v1_ | src_only;src_hit_but_op_name_unresolved |
| aclnnForeachMaximumScalarList | 已接入 | src_scan | _foreach_maximum.List;_foreach_maximum.Scalar;_foreach_maximum.ScalarList;_foreach_maximum_.List;_foreach_maximum_.Scalar;_foreach_maximum_.ScalarList |  | True | _foreach_maximum;_foreach_maximum_;_split_and_exec_npu_cmd_max_scalar_list | shared_by_6_ops;src_only |
| aclnnForeachMaximumScalarV2 | 已接入 | src_scan | _foreach_maximum.List;_foreach_maximum.Scalar;_foreach_maximum.ScalarList;_foreach_maximum_.List;_foreach_maximum_.Scalar;_foreach_maximum_.ScalarList |  | True | _foreach_maximum;_foreach_maximum_;_split_and_exec_npu_cmd_max_scalar | shared_by_6_ops;src_only |
| aclnnForeachMinimumList | 已接入 | src_scan | _foreach_minimum.List;_foreach_minimum.Scalar;_foreach_minimum.ScalarList;_foreach_minimum_.List;_foreach_minimum_.Scalar;_foreach_minimum_.ScalarList |  | True | _foreach_minimum;_foreach_minimum_;_split_and_exec_npu_cmd_min | shared_by_6_ops;src_only |
| aclnnForeachMinimumScalar | 已接入 | src_scan |  |  | True | _foreach_minimum_v1;_foreach_minimum_v1_ | src_only;src_hit_but_op_name_unresolved |
| aclnnForeachMinimumScalarList | 已接入 | src_scan | _foreach_minimum.List;_foreach_minimum.Scalar;_foreach_minimum.ScalarList;_foreach_minimum_.List;_foreach_minimum_.Scalar;_foreach_minimum_.ScalarList |  | True | _foreach_minimum;_foreach_minimum_;_split_and_exec_npu_cmd_min_scalar_list | shared_by_6_ops;src_only |
| aclnnForeachMinimumScalarV2 | 已接入 | src_scan | _foreach_minimum.List;_foreach_minimum.Scalar;_foreach_minimum.ScalarList;_foreach_minimum_.List;_foreach_minimum_.Scalar;_foreach_minimum_.ScalarList |  | True | _foreach_minimum;_foreach_minimum_;_split_and_exec_npu_cmd_min_scalar | shared_by_6_ops;src_only |
| aclnnForeachMulList | 已接入 | src_scan | _foreach_mul.List;_foreach_mul.Scalar;_foreach_mul.ScalarList;_foreach_mul_.List;_foreach_mul_.Scalar;_foreach_mul_.ScalarList |  | True | _foreach_mul;_foreach_mul_;_split_and_exec_npu_cmd_mul | shared_by_6_ops;src_only |
| aclnnForeachMulScalar | 已接入 | src_scan |  |  | True | _foreach_mul_v1;_foreach_mul_v1_ | src_only;src_hit_but_op_name_unresolved |
| aclnnForeachMulScalarList | 已接入 | src_scan | _foreach_mul.List;_foreach_mul.Scalar;_foreach_mul.ScalarList;_foreach_mul_.List;_foreach_mul_.Scalar;_foreach_mul_.ScalarList |  | True | _foreach_mul;_foreach_mul_;_split_and_exec_npu_cmd_mul_scalarlist | shared_by_6_ops;src_only |
| aclnnForeachMulScalarV2 | 已接入 | src_scan | _foreach_mul.List;_foreach_mul.Scalar;_foreach_mul.ScalarList;_foreach_mul_.List;_foreach_mul_.Scalar;_foreach_mul_.ScalarList |  | True | _foreach_mul;_foreach_mul_;_split_and_exec_npu_cmd_mul | shared_by_6_ops;src_only |
| aclnnForeachNeg | 已接入 | src_scan | _foreach_neg;_foreach_neg_ | _foreach_neg | False | _foreach_neg;_foreach_neg_;_split_and_exec_npu_cmd_neg | shared_by_2_ops;src_only |
| aclnnForeachNonFiniteCheckAndUnscale | 已接入 | src_scan |  |  | True | _split_and_exec_npu_cmd_ | src_only;src_hit_but_op_name_unresolved |
| aclnnForeachNorm | 已接入 | src_scan | _foreach_norm.Scalar | _foreach_norm.Scalar | False | _foreach_norm;_split_and_exec_npu_cmd_norm | src_only |
| aclnnForeachPowList | 已接入 | src_scan | _foreach_pow.List;_foreach_pow.Scalar;_foreach_pow.ScalarAndTensor;_foreach_pow.ScalarList;_foreach_pow_.List;_foreach_pow_.Scalar;_foreach_pow_.ScalarList |  | True | _foreach_pow;_foreach_pow_;_split_and_exec_npu_cmd_pow | shared_by_7_ops;src_only |
| aclnnForeachPowScalar | 已接入 | src_scan |  |  | True | _foreach_pow_v1;_foreach_pow_v1_ | src_only;src_hit_but_op_name_unresolved |
| aclnnForeachPowScalarAndTensor | 已接入 | src_scan | _foreach_pow.List;_foreach_pow.Scalar;_foreach_pow.ScalarAndTensor;_foreach_pow.ScalarList |  | True | _foreach_pow;_split_and_exec_npu_cmd_pow_scalar | shared_by_4_ops;src_only |
| aclnnForeachPowScalarList | 已接入 | src_scan | _foreach_pow.List;_foreach_pow.Scalar;_foreach_pow.ScalarAndTensor;_foreach_pow.ScalarList;_foreach_pow_.List;_foreach_pow_.Scalar;_foreach_pow_.ScalarList |  | True | _foreach_pow;_foreach_pow_;_split_and_exec_npu_cmd_pow_scalarlist | shared_by_7_ops;src_only |
| aclnnForeachPowScalarV2 | 已接入 | src_scan | _foreach_pow.List;_foreach_pow.Scalar;_foreach_pow.ScalarAndTensor;_foreach_pow.ScalarList;_foreach_pow_.List;_foreach_pow_.Scalar;_foreach_pow_.ScalarList |  | True | _foreach_pow;_foreach_pow_;_split_and_exec_npu_cmd_pow_kernel | shared_by_7_ops;src_only |
| aclnnForeachReciprocal | 已接入 | src_scan | _foreach_reciprocal;_foreach_reciprocal_ | _foreach_reciprocal | False | _foreach_reciprocal;_foreach_reciprocal_;_split_and_exec_npu_cmd_reciprocal | shared_by_2_ops;src_only |
| aclnnForeachRoundOffNumber | 已接入 | src_scan |  |  | True | exec_npu_cmd_v2;exec_npu_cmd_v2_ | src_only;src_hit_but_op_name_unresolved |
| aclnnForeachRoundOffNumberV2 | 已接入 | src_scan | _foreach_ceil;_foreach_ceil_;_foreach_floor;_foreach_floor_;_foreach_frac;_foreach_frac_;_foreach_round;_foreach_round_;_foreach_trunc;_foreach_trunc_ |  | True | _foreach_ceil;_foreach_ceil_;_foreach_floor;_foreach_floor_;_foreach_frac;_foreach_frac_;_foreach_round;_foreach_round_;...(+3) | shared_by_10_ops;src_only |
| aclnnForeachSigmoid | 已接入 | src_scan | _foreach_sigmoid;_foreach_sigmoid_ | _foreach_sigmoid | False | _foreach_sigmoid;_foreach_sigmoid_;_split_and_exec_npu_cmd_sigmoid | shared_by_2_ops;src_only |
| aclnnForeachSign | 已接入 | src_scan | _foreach_sign;_foreach_sign_ | _foreach_sign | False | _foreach_sign;_foreach_sign_;_split_and_exec_npu_cmd_sign | shared_by_2_ops;src_only |
| aclnnForeachSin | 已接入 | src_scan | _foreach_sin;_foreach_sin_ | _foreach_sin | False | _foreach_sin;_foreach_sin_;_split_and_exec_npu_cmd_sin | shared_by_2_ops;src_only |
| aclnnForeachSinh | 已接入 | src_scan | _foreach_sinh;_foreach_sinh_ | _foreach_sinh | False | _foreach_sinh;_foreach_sinh_;_split_and_exec_npu_cmd_sinh | shared_by_2_ops;src_only |
| aclnnForeachSqrt | 已接入 | src_scan | _foreach_sqrt;_foreach_sqrt_ | _foreach_sqrt | False | _foreach_sqrt;_foreach_sqrt_;_split_and_exec_npu_cmd_sqrt | shared_by_2_ops;src_only |
| aclnnForeachSubList | 已接入 | src_scan |  |  | True | _foreach_sub_v1;_foreach_sub_v1_ | src_only;src_hit_but_op_name_unresolved |
| aclnnForeachSubListV2 | 已接入 | src_scan | _foreach_sub.List;_foreach_sub.Scalar;_foreach_sub.ScalarList;_foreach_sub_.List;_foreach_sub_.Scalar;_foreach_sub_.ScalarList |  | True | _foreach_sub;_foreach_sub_;_split_and_exec_npu_cmd_sub | shared_by_6_ops;src_only |
| aclnnForeachSubScalar | 已接入 | src_scan |  |  | True | _foreach_sub_v1;_foreach_sub_v1_ | src_only;src_hit_but_op_name_unresolved |
| aclnnForeachSubScalarList | 已接入 | src_scan | _foreach_sub.List;_foreach_sub.Scalar;_foreach_sub.ScalarList;_foreach_sub_.List;_foreach_sub_.Scalar;_foreach_sub_.ScalarList |  | True | _foreach_sub;_foreach_sub_;_split_and_exec_npu_cmd_sub_scalarlist | shared_by_6_ops;src_only |
| aclnnForeachSubScalarV2 | 已接入 | src_scan | _foreach_sub.List;_foreach_sub.Scalar;_foreach_sub.ScalarList;_foreach_sub_.List;_foreach_sub_.Scalar;_foreach_sub_.ScalarList |  | True | _foreach_sub;_foreach_sub_;_split_and_exec_npu_cmd_sub_scalar | shared_by_6_ops;src_only |
| aclnnForeachTan | 已接入 | src_scan | _foreach_tan;_foreach_tan_ | _foreach_tan | False | _foreach_tan;_foreach_tan_;_split_and_exec_npu_cmd_tan | shared_by_2_ops;src_only |
| aclnnForeachTanh | 已接入 | src_scan | _foreach_tanh;_foreach_tanh_ | _foreach_tanh | False | _foreach_tanh;_foreach_tanh_;_split_and_exec_npu_cmd_tanh | shared_by_2_ops;src_only |
| aclnnForeachZeroInplace | 已接入 | src_scan | _foreach_zero_ |  | True | _foreach_zero_;_split_and_exec_npu_cmd_zero | src_only |
| aclnnFrac | 已接入 | yaml_exec | frac;frac.out | frac;frac.out | False |  | shared_by_2_ops;yaml_only |
| aclnnFusedCrossEntropyLossWithMaxSum | 已接入 | src_scan | fused_cross_entropy_loss_with_max_sum | fused_cross_entropy_loss_with_max_sum | False | fused_cross_entropy_loss_with_max_sum | src_only |
| aclnnFusedInferAttentionScore | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFusedInferAttentionScoreV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFusedInferAttentionScoreV3 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFusedInferAttentionScoreV4 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnFusedLinearCrossEntropyLossGrad | 已接入 | src_scan | fused_linear_cross_entropy_loss_with_max_sum_grad |  | True | fused_linear_cross_entropy_loss_with_max_sum_grad | src_only |
| aclnnFusedLinearOnlineMaxSum | 已接入 | src_scan | fused_linear_online_max_sum | fused_linear_online_max_sum | False | fused_linear_online_max_sum | src_only |
| aclnnGather | 已接入 | src_scan | gather;gather.dimname;gather.dimname_out;gather.out | gather;gather.dimname;gather.dimname_out;gather.out | False | gather;gather_out | shared_by_4_ops;src_only |
| aclnnGatherNd | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGatherPaKvCache | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGatherV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGatherV3 | 已接入 | yaml_exec | npu_gather_sparse_index |  | True |  | yaml_only |
| aclnnGcd | 已接入 | yaml_exec | gcd.out | gcd.out | False |  | yaml_only |
| aclnnGeGlu | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGeGluBackward | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGeGluV3 | 已接入 | src_scan | npu_geglu |  | True | npu_geglu | src_only |
| aclnnGeGluV3Backward | 已接入 | src_scan | npu_geglu_grad |  | True | npu_geglu_grad | src_only |
| aclnnGeScalar | 已接入 | src_scan | ge.Scalar;ge.Scalar_out;ge.Tensor;ge.Tensor_out |  | True | ge;ge_out | shared_by_4_ops;src_only |
| aclnnGeTensor | 已接入 | src_scan | ge.Scalar;ge.Scalar_out;ge.Tensor;ge.Tensor_out |  | True | ge;ge_out | shared_by_4_ops;src_only |
| aclnnGelu | 已接入 | src_scan | gelu.out | gelu.out | False | gelu_out | src_only |
| aclnnGeluBackward | 已接入 | src_scan | gelu_backward | gelu_backward | False | gelu_backward | src_only |
| aclnnGeluBackwardV2 | 已接入 | yaml_exec+src_scan | gelu_backward;npu_gelu_backward |  | True | gelu_backward | shared_by_2_ops;yaml+src |
| aclnnGeluMul | 已接入 | yaml_exec | npu_gelu_mul |  | True |  | yaml_only |
| aclnnGeluV2 | 已接入 | yaml_exec+src_scan | gelu.out;npu_gelu |  | True | gelu_out | shared_by_2_ops;yaml+src |
| aclnnGemm | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGemmaRmsNorm | 已接入 | src_scan | npu_gemma_rms_norm |  | True | npu_gemma_rms_norm | src_only |
| aclnnGer | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGlobalAveragePool | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGlobalMaxPool | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGlu | 已接入 | yaml_exec | glu;glu.out | glu;glu.out | False |  | shared_by_2_ops;yaml_only |
| aclnnGluBackward | 已接入 | yaml_exec | glu_backward;glu_backward.grad_input | glu_backward;glu_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnGridSampler2D | 已接入 | yaml_exec | grid_sampler_2d |  | True |  | yaml_only |
| aclnnGridSampler2DBackward | 已接入 | yaml_exec | grid_sampler_2d_backward |  | True |  | yaml_only |
| aclnnGridSampler3D | 已接入 | yaml_exec | grid_sampler_3d |  | True |  | yaml_only |
| aclnnGridSampler3DBackward | 已接入 | yaml_exec | grid_sampler_3d_backward |  | True |  | yaml_only |
| aclnnGroupNorm | 已接入 | yaml_exec | native_group_norm |  | True |  | yaml_only |
| aclnnGroupNormBackward | 已接入 | src_scan | native_group_norm_backward |  | True | native_group_norm_backward | src_only |
| aclnnGroupNormSilu | 已接入 | yaml_exec | npu_group_norm_silu |  | True |  | yaml_only |
| aclnnGroupNormSiluV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGroupNormSwish | 已接入 | yaml_exec | npu_group_norm_swish |  | True |  | yaml_only |
| aclnnGroupNormSwishGrad | 已接入 | yaml_exec | npu_group_norm_swish_grad |  | True |  | yaml_only |
| aclnnGroupQuant | 已接入 | yaml_exec | npu_group_quant |  | True |  | yaml_only |
| aclnnGroupedBiasAddGrad | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGroupedBiasAddGradV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGroupedMatMulAllReduce | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGroupedMatMulAlltoAllv | 已接入 | src_scan |  |  | True | options | src_only;src_hit_but_op_name_unresolved |
| aclnnGroupedMatmul | 已接入 | src_scan | npu_grouped_matmul;npu_grouped_matmul.List |  | True | npu_grouped_matmul | shared_by_2_ops;src_only |
| aclnnGroupedMatmulAdd | 已接入 | src_scan | npu_grouped_matmul_add |  | True | IsAclnnOnly;npu_grouped_matmul_add | src_only |
| aclnnGroupedMatmulFinalizeRouting | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGroupedMatmulFinalizeRoutingV2 | 已接入 | src_scan | npu_grouped_matmul_finalize_routing |  | True | npu_grouped_matmul_finalize_routing | src_only |
| aclnnGroupedMatmulFinalizeRoutingV3 | 已接入 | src_scan | npu_grouped_matmul_finalize_routing |  | True | npu_grouped_matmul_finalize_routing | src_only |
| aclnnGroupedMatmulFinalizeRoutingWeightNz | 已接入 | src_scan | npu_grouped_matmul_finalize_routing |  | True | npu_grouped_matmul_finalize_routing | src_only |
| aclnnGroupedMatmulFinalizeRoutingWeightNzV2 | 已接入 | src_scan | npu_grouped_matmul_finalize_routing |  | True | npu_grouped_matmul_finalize_routing | src_only |
| aclnnGroupedMatmulSwigluQuant | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGroupedMatmulSwigluQuantV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGroupedMatmulSwigluQuantWeightNZ | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGroupedMatmulV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGroupedMatmulV3 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnGroupedMatmulV4 | 已接入 | src_scan | npu_grouped_matmul;npu_grouped_matmul.List |  | True | npu_grouped_matmul | shared_by_2_ops;src_only |
| aclnnGroupedMatmulV5 | 已接入 | src_scan | npu_grouped_matmul;npu_grouped_matmul.List |  | True | npu_grouped_matmul | shared_by_2_ops;src_only |
| aclnnGroupedMatmulWeightNz | 已接入 | src_scan | npu_grouped_matmul;npu_grouped_matmul.List |  | True | npu_grouped_matmul | shared_by_2_ops;src_only |
| aclnnGtScalar | 已接入 | src_scan | gt.Scalar;gt.Scalar_out;gt.Tensor;gt.Tensor_out |  | True | gt;gt_out | shared_by_4_ops;src_only |
| aclnnGtTensor | 已接入 | src_scan | gt.Scalar;gt.Scalar_out;gt.Tensor;gt.Tensor_out |  | True | gt;gt_out | shared_by_4_ops;src_only |
| aclnnHardshrink | 已接入 | yaml_exec | hardshrink;hardshrink.out | hardshrink;hardshrink.out | False |  | shared_by_2_ops;yaml_only |
| aclnnHardshrinkBackward | 已接入 | yaml_exec | hardshrink_backward;hardshrink_backward.grad_input | hardshrink_backward;hardshrink_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnHardsigmoid | 已接入 | yaml_exec | hardsigmoid;hardsigmoid.out | hardsigmoid;hardsigmoid.out | False |  | shared_by_2_ops;yaml_only |
| aclnnHardsigmoidBackward | 已接入 | yaml_exec | hardsigmoid_backward | hardsigmoid_backward | False |  | yaml_only |
| aclnnHardswish | 已接入 | yaml_exec | hardswish;hardswish.out | hardswish;hardswish.out | False |  | shared_by_2_ops;yaml_only |
| aclnnHardswishBackward | 已接入 | src_scan | hardswish_backward | hardswish_backward | False | hardswish_backward | src_only |
| aclnnHardswishBackwardV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnHardtanh | 已接入 | yaml_exec | hardtanh;hardtanh.out | hardtanh;hardtanh.out | False |  | shared_by_2_ops;yaml_only |
| aclnnHardtanhBackward | 已接入 | yaml_exec | hardtanh_backward;hardtanh_backward.grad_input | hardtanh_backward;hardtanh_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnHeaviside | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnHistc | 已接入 | yaml_exec | histc;histc.out | histc;histc.out | False |  | shared_by_2_ops;yaml_only |
| aclnnIm2col | 已接入 | src_scan | im2col;im2col.out | im2col;im2col.out | False | im2col;im2col_out | shared_by_2_ops;src_only |
| aclnnIm2colBackward | 已接入 | yaml_exec | col2im;col2im.out |  | True |  | shared_by_2_ops;yaml_only |
| aclnnIncreFlashAttention | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnIncreFlashAttentionV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnIncreFlashAttentionV3 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnIncreFlashAttentionV4 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnIndex | 已接入 | src_scan | index.Tensor | index.Tensor | False | index;index_high_dims_op_api | src_only |
| aclnnIndexAdd | 已接入 | src_scan | index_add;index_add.dimname;index_add.out | index_add;index_add.dimname;index_add.out | False | check_tensor;index_add | shared_by_3_ops;src_only |
| aclnnIndexCopy | 已接入 | yaml_exec | index_copy;index_copy.out | index_copy;index_copy.out | False |  | shared_by_2_ops;yaml_only |
| aclnnIndexFill | 已接入 | src_scan | index_fill.int_Scalar;index_fill.int_Tensor | index_fill.int_Scalar;index_fill.int_Tensor | False | index_fill | shared_by_2_ops;src_only |
| aclnnIndexFillTensor | 已接入 | src_scan | index_fill.int_Scalar;index_fill.int_Tensor |  | True | index_fill | shared_by_2_ops;src_only |
| aclnnIndexPutImpl | 已接入 | src_scan | _index_put_impl_;index_put;index_put_ |  | True | _index_put_impl_;index_put;index_put_ | shared_by_3_ops;src_only |
| aclnnIndexSelect | 已接入 | src_scan | index_select;index_select.dimname;index_select.dimname_out;index_select.out | index_select;index_select.dimname;index_select.dimname_out;index_select.out | False | index_select;index_select_out | shared_by_4_ops;src_only |
| aclnnInit | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnInplaceAcos | 已接入 | yaml_exec | acos_ | acos_ | False |  | yaml_only |
| aclnnInplaceAcosh | 已接入 | yaml_exec | acosh_ | acosh_ | False |  | yaml_only |
| aclnnInplaceAdd | 已接入 | src_scan | add_.Scalar;add_.Tensor | add_.Scalar;add_.Tensor | False | add_;inplace_add_out_npu_no_check | shared_by_2_ops;src_only |
| aclnnInplaceAddRelu | 已接入 | yaml_exec | _add_relu_.Tensor | _add_relu_.Tensor | False |  | yaml_only |
| aclnnInplaceAddRmsNorm | 已接入 | src_scan | npu_add_rms_norm_v2;npu_add_rms_norm_v2_functional |  | True | npu_add_rms_norm_v2;npu_add_rms_norm_v2_functional | shared_by_2_ops;src_only |
| aclnnInplaceAddbmm | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnInplaceAddcdiv | 已接入 | src_scan | addcdiv_ | addcdiv_ | False | addcdiv_ | src_only |
| aclnnInplaceAddcmul | 已接入 | src_scan | addcmul_ | addcmul_ | False | addcmul_ | src_only |
| aclnnInplaceAddmm | 已接入 | src_scan | addmm_ | addmm_ | False | addmm_ | src_only |
| aclnnInplaceAddr | 已接入 | yaml_exec | addr_ | addr_ | False |  | yaml_only |
| aclnnInplaceAdds | 已接入 | src_scan | add_.Scalar;add_.Tensor |  | True | add_;inplace_add_out_npu_no_check | shared_by_2_ops;src_only |
| aclnnInplaceAsin | 已接入 | yaml_exec | asin_ | asin_ | False |  | yaml_only |
| aclnnInplaceAsinh | 已接入 | yaml_exec | asinh_ | asinh_ | False |  | yaml_only |
| aclnnInplaceAtan | 已接入 | yaml_exec | atan_ | atan_ | False |  | yaml_only |
| aclnnInplaceAtan2 | 已接入 | src_scan | atan2_ | atan2_ | False | atan2_ | src_only |
| aclnnInplaceAtanh | 已接入 | yaml_exec | atanh_ | atanh_ | False |  | yaml_only |
| aclnnInplaceAttentionWorkerScheduler | 已接入 | yaml_exec+src_scan | attention_worker_scheduler;attention_worker_scheduler_ | attention_worker_scheduler_ | False | attention_worker_scheduler | shared_by_2_ops;yaml+src |
| aclnnInplaceBaddbmm | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnInplaceBernoulli | 已接入 | src_scan | bernoulli;bernoulli.out;bernoulli.p;bernoulli_.Tensor;bernoulli_.float | bernoulli_.Tensor;bernoulli_.float | False | bernoulli;bernoulli_ | shared_by_5_ops;src_only |
| aclnnInplaceBernoulliTensor | 已接入 | src_scan | bernoulli;bernoulli.out;bernoulli.p;bernoulli_.Tensor;bernoulli_.float |  | True | bernoulli;bernoulli_;bernoulli_out | shared_by_5_ops;src_only |
| aclnnInplaceBitwiseAndScalar | 已接入 | src_scan | bitwise_and_.Scalar;bitwise_and_.Tensor |  | True | bitwise_and_;bitwise_and_inplace_op_api_out_npu_nocheck | shared_by_2_ops;src_only |
| aclnnInplaceBitwiseAndTensor | 已接入 | src_scan | bitwise_and_.Scalar;bitwise_and_.Tensor |  | True | bitwise_and_;bitwise_and_inplace_op_api_out_npu_nocheck | shared_by_2_ops;src_only |
| aclnnInplaceBitwiseOrScalar | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnInplaceBitwiseOrTensor | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnInplaceBitwiseXorScalar | 已接入 | src_scan | bitwise_xor_.Scalar;bitwise_xor_.Tensor |  | True | bitwise_xor_ | shared_by_2_ops;src_only |
| aclnnInplaceBitwiseXorTensor | 已接入 | src_scan | bitwise_xor_.Scalar;bitwise_xor_.Tensor |  | True | bitwise_xor_ | shared_by_2_ops;src_only |
| aclnnInplaceCeil | 已接入 | yaml_exec | ceil_ | ceil_ | False |  | yaml_only |
| aclnnInplaceCelu | 已接入 | yaml_exec | celu_ | celu_ | False |  | yaml_only |
| aclnnInplaceClampMax | 已接入 | yaml_exec | clamp_max_ | clamp_max_ | False |  | yaml_only |
| aclnnInplaceClampMaxTensor | 已接入 | yaml_exec | clamp_max_.Tensor |  | True |  | yaml_only |
| aclnnInplaceClampMinTensor | 已接入 | yaml_exec | clamp_min_.Tensor |  | True |  | yaml_only |
| aclnnInplaceCopy | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnInplaceCos | 已接入 | yaml_exec | cos_ | cos_ | False |  | yaml_only |
| aclnnInplaceCosh | 已接入 | yaml_exec | cosh_ | cosh_ | False |  | yaml_only |
| aclnnInplaceCumprod | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnInplaceDiv | 已接入 | src_scan | div_.Scalar;div_.Scalar_mode;div_.Tensor;div_.Tensor_mode | div_.Scalar;div_.Scalar_mode;div_.Tensor;div_.Tensor_mode | False | div_;inplace_div_out_npu_no_check | shared_by_4_ops;src_only |
| aclnnInplaceDivMod | 已接入 | src_scan | div_.Scalar;div_.Scalar_mode;div_.Tensor;div_.Tensor_mode |  | True | div_;inplace_div_out_mode_npu_no_check | shared_by_4_ops;src_only |
| aclnnInplaceDivMods | 已接入 | src_scan | div_.Scalar;div_.Scalar_mode;div_.Tensor;div_.Tensor_mode |  | True | div_;inplace_div_out_mode_npu_no_check | shared_by_4_ops;src_only |
| aclnnInplaceDivs | 已接入 | src_scan | div_.Scalar;div_.Scalar_mode;div_.Tensor;div_.Tensor_mode |  | True | div_;inplace_div_out_npu_no_check | shared_by_4_ops;src_only |
| aclnnInplaceElu | 已接入 | yaml_exec | elu_ | elu_ | False |  | yaml_only |
| aclnnInplaceEqScalar | 已接入 | src_scan | eq_.Scalar;eq_.Tensor |  | True | eq_ | shared_by_2_ops;src_only |
| aclnnInplaceEqTensor | 已接入 | src_scan | eq_.Scalar;eq_.Tensor |  | True | eq_ | shared_by_2_ops;src_only |
| aclnnInplaceErf | 已接入 | yaml_exec | erf_ | erf_ | False |  | yaml_only |
| aclnnInplaceErfc | 已接入 | yaml_exec | erfc_ | erfc_ | False |  | yaml_only |
| aclnnInplaceErfinv | 已接入 | yaml_exec | erfinv_ | erfinv_ | False |  | yaml_only |
| aclnnInplaceExp | 已接入 | yaml_exec | exp_ | exp_ | False |  | yaml_only |
| aclnnInplaceExp2 | 已接入 | yaml_exec | exp2_ | exp2_ | False |  | yaml_only |
| aclnnInplaceExpm1 | 已接入 | yaml_exec | expm1_ | expm1_ | False |  | yaml_only |
| aclnnInplaceFfnWorkerScheduler | 已接入 | yaml_exec+src_scan | ffn_worker_scheduler;ffn_worker_scheduler_ | ffn_worker_scheduler_ | False | ffn_worker_scheduler | shared_by_2_ops;yaml+src |
| aclnnInplaceFillDiagonal | 已接入 | yaml_exec | fill_diagonal_ | fill_diagonal_ | False |  | yaml_only |
| aclnnInplaceFillScalar | 已接入 | src_scan | fill_.Scalar;fill_.Tensor |  | True | fill_ | shared_by_2_ops;src_only |
| aclnnInplaceFillTensor | 已接入 | src_scan | fill_.Scalar;fill_.Tensor |  | True | fill_ | shared_by_2_ops;src_only |
| aclnnInplaceFloor | 已接入 | yaml_exec | floor_ | floor_ | False |  | yaml_only |
| aclnnInplaceFloorDivide | 已接入 | src_scan | floor_divide_.Scalar;floor_divide_.Tensor | floor_divide_.Scalar;floor_divide_.Tensor | False | floor_divide_;inplace_floor_divide_out_npu_opapi | shared_by_2_ops;src_only |
| aclnnInplaceFloorDivides | 已接入 | src_scan | floor_divide_.Scalar;floor_divide_.Tensor |  | True | floor_divide_;inplace_floor_divide_out_npu_opapi | shared_by_2_ops;src_only |
| aclnnInplaceFmodScalar | 已接入 | yaml_exec | fmod_.Scalar |  | True |  | yaml_only |
| aclnnInplaceFmodTensor | 已接入 | yaml_exec | fmod_.Tensor |  | True |  | yaml_only |
| aclnnInplaceFrac | 已接入 | yaml_exec | frac_ | frac_ | False |  | yaml_only |
| aclnnInplaceGeScalar | 已接入 | src_scan | ge_.Scalar;ge_.Tensor |  | True | ge_ | shared_by_2_ops;src_only |
| aclnnInplaceGeTensor | 已接入 | src_scan | ge_.Scalar;ge_.Tensor |  | True | ge_ | shared_by_2_ops;src_only |
| aclnnInplaceGtScalar | 已接入 | src_scan | gt_.Scalar;gt_.Tensor |  | True | gt_ | shared_by_2_ops;src_only |
| aclnnInplaceGtTensor | 已接入 | src_scan | gt_.Scalar;gt_.Tensor |  | True | gt_ | shared_by_2_ops;src_only |
| aclnnInplaceHardsigmoid | 已接入 | yaml_exec | hardsigmoid_ | hardsigmoid_ | False |  | yaml_only |
| aclnnInplaceHardswish | 已接入 | yaml_exec | hardswish_ | hardswish_ | False |  | yaml_only |
| aclnnInplaceHardtanh | 已接入 | yaml_exec | hardtanh_ | hardtanh_ | False |  | yaml_only |
| aclnnInplaceIndexCopy | 已接入 | yaml_exec | index_copy_ | index_copy_ | False |  | yaml_only |
| aclnnInplaceIndexFill | 已接入 | src_scan | index_fill_.int_Scalar;index_fill_.int_Tensor | index_fill_.int_Scalar;index_fill_.int_Tensor | False | index_fill_ | shared_by_2_ops;src_only |
| aclnnInplaceIndexFillTensor | 已接入 | src_scan | index_fill_.int_Scalar;index_fill_.int_Tensor |  | True | index_fill_ | shared_by_2_ops;src_only |
| aclnnInplaceLeScalar | 已接入 | src_scan | le_.Scalar;le_.Tensor |  | True | le_ | shared_by_2_ops;src_only |
| aclnnInplaceLeTensor | 已接入 | src_scan | le_.Scalar;le_.Tensor |  | True | le_ | shared_by_2_ops;src_only |
| aclnnInplaceLeakyRelu | 已接入 | yaml_exec | leaky_relu_ | leaky_relu_ | False |  | yaml_only |
| aclnnInplaceLerp | 已接入 | yaml_exec | lerp_.Tensor | lerp_.Tensor | False |  | yaml_only |
| aclnnInplaceLerps | 已接入 | yaml_exec | lerp_.Scalar |  | True |  | yaml_only |
| aclnnInplaceLog | 已接入 | src_scan | log_ | log_ | False | log_ | src_only |
| aclnnInplaceLog10 | 已接入 | yaml_exec | log10_ | log10_ | False |  | yaml_only |
| aclnnInplaceLog1p | 已接入 | yaml_exec | log1p_ | log1p_ | False |  | yaml_only |
| aclnnInplaceLog2 | 已接入 | yaml_exec | log2_ | log2_ | False |  | yaml_only |
| aclnnInplaceLogicalAnd | 已接入 | yaml_exec | logical_and_ | logical_and_ | False |  | yaml_only |
| aclnnInplaceLogicalNot | 已接入 | yaml_exec | logical_not_ | logical_not_ | False |  | yaml_only |
| aclnnInplaceLogicalOr | 已接入 | src_scan | logical_or_ | logical_or_ | False | logical_or_ | src_only |
| aclnnInplaceLtScalar | 已接入 | src_scan | lt_.Scalar;lt_.Tensor |  | True | lt_ | shared_by_2_ops;src_only |
| aclnnInplaceLtTensor | 已接入 | src_scan | lt_.Scalar;lt_.Tensor |  | True | lt_ | shared_by_2_ops;src_only |
| aclnnInplaceMaskedFillScalar | 已接入 | src_scan | masked_fill_.Scalar;masked_fill_.Tensor |  | True | masked_fill_ | shared_by_2_ops;src_only |
| aclnnInplaceMaskedFillTensor | 已接入 | src_scan |  |  | True | IsCPUScalar | src_only;src_hit_but_op_name_unresolved |
| aclnnInplaceMaskedScatter | 已接入 | src_scan |  |  | True | check_memory | src_only;src_hit_but_op_name_unresolved |
| aclnnInplaceMatmulAllReduceAddRmsNorm | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnInplaceMish | 已接入 | src_scan | mish_ | mish_ | False | mish_ | src_only |
| aclnnInplaceMul | 已接入 | src_scan | mul_.Scalar;mul_.Tensor | mul_.Scalar;mul_.Tensor | False | inplace_mul_out_npu_no_check;mul_ | shared_by_2_ops;src_only |
| aclnnInplaceMuls | 已接入 | src_scan | mul_.Scalar;mul_.Tensor |  | True | inplace_mul_out_npu_no_check;mul_ | shared_by_2_ops;src_only |
| aclnnInplaceNanToNum | 已接入 | src_scan | nan_to_num_ | nan_to_num_ | False | nan_to_num_ | src_only |
| aclnnInplaceNeScalar | 已接入 | src_scan | ne_.Scalar;ne_.Tensor |  | True | ne_ | shared_by_2_ops;src_only |
| aclnnInplaceNeTensor | 已接入 | src_scan | ne_.Scalar;ne_.Tensor |  | True | ne_ | shared_by_2_ops;src_only |
| aclnnInplaceNeg | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnInplaceNormal | 已接入 | src_scan | normal_ | normal_ | False | normal_ | src_only |
| aclnnInplaceNormalTensor | 已接入 | src_scan | normal_ |  | True | normal_ | src_only |
| aclnnInplaceOne | 已接入 | src_scan | one_;ones;ones.names;ones.out;ones_like | one_ | False | one_;ones;ones_like;ones_out | shared_by_5_ops;src_only |
| aclnnInplacePowTensorScalar | 已接入 | yaml_exec | pow_.Scalar |  | True |  | yaml_only |
| aclnnInplacePowTensorTensor | 已接入 | yaml_exec | pow_.Tensor |  | True |  | yaml_only |
| aclnnInplacePut | 已接入 | src_scan |  |  | True | check_memory | src_only;src_hit_but_op_name_unresolved |
| aclnnInplaceQuantMatmulAllReduceAddRmsNorm | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnInplaceQuantScatter | 已接入 | src_scan | npu_quant_scatter;npu_quant_scatter_ |  | True | npu_quant_scatter;npu_quant_scatter_ | shared_by_2_ops;src_only |
| aclnnInplaceRReluWithNoise | 已接入 | src_scan | rrelu_with_noise_ |  | True | rrelu_with_noise_ | src_only |
| aclnnInplaceRandom | 已接入 | src_scan | random_;random_.from;random_.to | random_;random_.from;random_.to | False | random_;random_op_api_ | shared_by_3_ops;src_only |
| aclnnInplaceRandomTensor | 已接入 | src_scan |  |  | True | random_op_api_ | src_only;src_hit_but_op_name_unresolved |
| aclnnInplaceReciprocal | 已接入 | yaml_exec | reciprocal_ | reciprocal_ | False |  | yaml_only |
| aclnnInplaceRelu | 已接入 | yaml_exec | relu_ | relu_ | False |  | yaml_only |
| aclnnInplaceRemainderTensorScalar | 已接入 | src_scan | remainder_.Scalar;remainder_.Tensor |  | True | remainder_ | shared_by_2_ops;src_only |
| aclnnInplaceRemainderTensorTensor | 已接入 | src_scan | remainder_.Scalar;remainder_.Tensor |  | True | remainder_ | shared_by_2_ops;src_only |
| aclnnInplaceRenorm | 已接入 | src_scan | renorm_ | renorm_ | False | renorm_ | src_only |
| aclnnInplaceRound | 已接入 | yaml_exec | round_ | round_ | False |  | yaml_only |
| aclnnInplaceRoundDecimals | 已接入 | src_scan | round_;round_.decimals |  | True | round_ | shared_by_2_ops;src_only |
| aclnnInplaceRsqrt | 已接入 | yaml_exec | rsqrt_ | rsqrt_ | False |  | yaml_only |
| aclnnInplaceScatter | 已接入 | src_scan | scatter_.src;scatter_.value | scatter_.src;scatter_.value | False | scatter_ | shared_by_2_ops;src_only |
| aclnnInplaceScatterUpdate | 已接入 | src_scan | scatter_update_ | scatter_update_ | False | scatter_update_ | src_only |
| aclnnInplaceScatterValue | 已接入 | src_scan | scatter_.src;scatter_.value |  | True | scatter_ | shared_by_2_ops;src_only |
| aclnnInplaceSelu | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnInplaceSigmoid | 已接入 | yaml_exec | sigmoid_ | sigmoid_ | False |  | yaml_only |
| aclnnInplaceSin | 已接入 | yaml_exec | sin_ | sin_ | False |  | yaml_only |
| aclnnInplaceSinc | 已接入 | yaml_exec | sinc_ | sinc_ | False |  | yaml_only |
| aclnnInplaceSinh | 已接入 | yaml_exec | sinh_ | sinh_ | False |  | yaml_only |
| aclnnInplaceSqrt | 已接入 | yaml_exec | sqrt_ | sqrt_ | False |  | yaml_only |
| aclnnInplaceSub | 已接入 | src_scan | sub_.Scalar;sub_.Tensor | sub_.Scalar;sub_.Tensor | False | inplace_sub_out_npu_no_check;sub_ | shared_by_2_ops;src_only |
| aclnnInplaceSubs | 已接入 | src_scan | sub_.Scalar;sub_.Tensor |  | True | inplace_sub_out_npu_no_check;sub_ | shared_by_2_ops;src_only |
| aclnnInplaceTan | 已接入 | yaml_exec | tan_ | tan_ | False |  | yaml_only |
| aclnnInplaceTanh | 已接入 | src_scan | tanh_ | tanh_ | False | tanh_ | src_only |
| aclnnInplaceThreshold | 已接入 | src_scan | threshold_ | threshold_ | False | threshold_ | src_only |
| aclnnInplaceTril | 已接入 | yaml_exec | tril_ | tril_ | False |  | yaml_only |
| aclnnInplaceTriu | 已接入 | yaml_exec | triu_ | triu_ | False |  | yaml_only |
| aclnnInplaceTrunc | 已接入 | yaml_exec | trunc_ | trunc_ | False |  | yaml_only |
| aclnnInplaceUniform | 已接入 | src_scan | uniform_ | uniform_ | False | uniform_ | src_only |
| aclnnInplaceUniformTensor | 已接入 | src_scan | uniform_ |  | True | uniform_ | src_only |
| aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnInplaceXLogYScalarOther | 已接入 | yaml_exec | xlogy_.Scalar_Other |  | True |  | yaml_only |
| aclnnInplaceXLogYTensor | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnInplaceZero | 已接入 | src_scan | zero_;zeros;zeros.names;zeros.out | zero_ | False | zero_;zeros;zeros_out;zeros_symint | shared_by_4_ops;src_only |
| aclnnInstanceNorm | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnInverse | 已接入 | yaml_exec | inverse;inverse.out | inverse;inverse.out | False |  | shared_by_2_ops;yaml_only |
| aclnnIou | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnIsClose | 已接入 | yaml_exec | isclose |  | True |  | yaml_only |
| aclnnIsFinite | 已接入 | yaml_exec | isfinite |  | True |  | yaml_only |
| aclnnIsInScalarTensor | 已接入 | yaml_exec | isin.Scalar_Tensor_out |  | True |  | yaml_only |
| aclnnIsInTensorScalar | 已接入 | yaml_exec | isin.Tensor_Scalar;isin.Tensor_Scalar_out |  | True |  | shared_by_2_ops;yaml_only |
| aclnnIsInf | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnIsNegInf | 已接入 | yaml_exec | isneginf.out |  | True |  | yaml_only |
| aclnnIsPosInf | 已接入 | yaml_exec | isposinf.out |  | True |  | yaml_only |
| aclnnKlDiv | 已接入 | yaml_exec | kl_div | kl_div | False |  | yaml_only |
| aclnnKlDivBackward | 已接入 | yaml_exec | kl_div_backward | kl_div_backward | False |  | yaml_only |
| aclnnKthvalue | 已接入 | src_scan | kthvalue;kthvalue.dimname;kthvalue.dimname_out;kthvalue.values | kthvalue;kthvalue.dimname;kthvalue.dimname_out;kthvalue.values | False | kthvalue;kthvalue_out | shared_by_4_ops;src_only |
| aclnnL1Loss | 已接入 | yaml_exec | l1_loss | l1_loss | False |  | yaml_only |
| aclnnL1LossBackward | 已接入 | yaml_exec | l1_loss_backward | l1_loss_backward | False |  | yaml_only |
| aclnnLayerNorm | 已接入 | src_scan |  |  | True | Tensor | src_only;src_hit_but_op_name_unresolved |
| aclnnLayerNormBackward | 已接入 | src_scan | native_layer_norm_backward |  | True | native_layer_norm_backward | src_only |
| aclnnLayerNormWithImplMode | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnLeScalar | 已接入 | src_scan | le.Scalar;le.Scalar_out;le.Tensor;le.Tensor_out |  | True | le;le_out | shared_by_4_ops;src_only |
| aclnnLeTensor | 已接入 | src_scan | le.Scalar;le.Scalar_out;le.Tensor;le.Tensor_out |  | True | le;scalar_type | shared_by_4_ops;src_only |
| aclnnLeakyRelu | 已接入 | yaml_exec | leaky_relu;leaky_relu.out | leaky_relu;leaky_relu.out | False |  | shared_by_2_ops;yaml_only |
| aclnnLeakyReluBackward | 已接入 | yaml_exec | leaky_relu_backward;leaky_relu_backward.grad_input | leaky_relu_backward;leaky_relu_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnLerp | 已接入 | yaml_exec | lerp.Tensor;lerp.Tensor_out | lerp.Tensor;lerp.Tensor_out | False |  | shared_by_2_ops;yaml_only |
| aclnnLerps | 已接入 | yaml_exec | lerp.Scalar;lerp.Scalar_out |  | True |  | shared_by_2_ops;yaml_only |
| aclnnLgamma | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnLightningIndexerGrad | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnLinalgCholesky | 已接入 | src_scan | linalg_cholesky;linalg_cholesky.out | linalg_cholesky;linalg_cholesky.out | False | linalg_cholesky;linalg_cholesky_out | shared_by_2_ops;src_only |
| aclnnLinalgCross | 已接入 | yaml_exec | linalg_cross;linalg_cross.out | linalg_cross;linalg_cross.out | False |  | shared_by_2_ops;yaml_only |
| aclnnLinalgQr | 已接入 | src_scan | linalg_qr;linalg_qr.out | linalg_qr;linalg_qr.out | False | linalg_qr;linalg_qr_out | shared_by_2_ops;src_only |
| aclnnLinalgVectorNorm | 已接入 | src_scan | linalg_vector_norm;linalg_vector_norm.out | linalg_vector_norm;linalg_vector_norm.out | False | linalg_vector_norm;linalg_vector_norm_out | shared_by_2_ops;src_only |
| aclnnLinspace | 已接入 | src_scan | linspace;linspace.out | linspace;linspace.out | False | linspace;linspace_out | shared_by_2_ops;src_only |
| aclnnLog | 已接入 | src_scan | log;log.out | log;log.out | False | check_memory;log | shared_by_2_ops;src_only |
| aclnnLog10 | 已接入 | yaml_exec | log10;log10.out | log10;log10.out | False |  | shared_by_2_ops;yaml_only |
| aclnnLog1p | 已接入 | yaml_exec | log1p;log1p.out | log1p;log1p.out | False |  | shared_by_2_ops;yaml_only |
| aclnnLog2 | 已接入 | yaml_exec | log2;log2.out | log2;log2.out | False |  | shared_by_2_ops;yaml_only |
| aclnnLogAddExp | 已接入 | yaml_exec | logaddexp;logaddexp.out |  | True |  | shared_by_2_ops;yaml_only |
| aclnnLogAddExp2 | 已接入 | yaml_exec | logaddexp2;logaddexp2.out |  | True |  | shared_by_2_ops;yaml_only |
| aclnnLogSigmoid | 已接入 | src_scan | log_sigmoid;log_sigmoid.out;log_sigmoid_forward;log_sigmoid_forward.output | log_sigmoid;log_sigmoid.out | False | log_sigmoid;log_sigmoid_forward;log_sigmoid_out | shared_by_4_ops;src_only |
| aclnnLogSigmoidBackward | 已接入 | yaml_exec | log_sigmoid_backward;log_sigmoid_backward.grad_input | log_sigmoid_backward;log_sigmoid_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnLogSigmoidForward | 已接入 | src_scan | log_sigmoid_forward;log_sigmoid_forward.output | log_sigmoid_forward;log_sigmoid_forward.output | False | check_tensor;log_sigmoid_forward | shared_by_2_ops;src_only |
| aclnnLogSoftmax | 已接入 | yaml_exec | _log_softmax | _log_softmax | False |  | yaml_only |
| aclnnLogSoftmaxBackward | 已接入 | yaml_exec | _log_softmax_backward_data;_log_softmax_backward_data.out |  | True |  | shared_by_2_ops;yaml_only |
| aclnnLogSumExp | 已接入 | src_scan | logsumexp;logsumexp.names;logsumexp.names_out;logsumexp.out |  | True | logsumexp;scalar_type | shared_by_4_ops;src_only |
| aclnnLogdet | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnLogicalAnd | 已接入 | yaml_exec | logical_and;logical_and.out | logical_and;logical_and.out | False |  | shared_by_2_ops;yaml_only |
| aclnnLogicalNot | 已接入 | yaml_exec | logical_not;logical_not.out | logical_not;logical_not.out | False |  | shared_by_2_ops;yaml_only |
| aclnnLogicalOr | 已接入 | src_scan | logical_or;logical_or.out | logical_or;logical_or.out | False | logical_or;logical_or_out | shared_by_2_ops;src_only |
| aclnnLogicalXor | 已接入 | yaml_exec | logical_xor;logical_xor.out | logical_xor;logical_xor.out | False |  | shared_by_2_ops;yaml_only |
| aclnnLogit | 已接入 | src_scan | logit;logit.out | logit;logit.out | False | logit;logit_out | shared_by_2_ops;src_only |
| aclnnLogitGrad | 已接入 | src_scan | logit_backward;logit_backward.grad_input |  | True | logit_backward;logit_backward_out | shared_by_2_ops;src_only |
| aclnnLtScalar | 已接入 | src_scan | lt.Scalar;lt.Scalar_out;lt.Tensor;lt.Tensor_out |  | True | lt;lt_out | shared_by_4_ops;src_only |
| aclnnLtTensor | 已接入 | src_scan | lt.Scalar;lt.Scalar_out;lt.Tensor;lt.Tensor_out |  | True | lt;lt_out | shared_by_4_ops;src_only |
| aclnnMaskedSelect | 已接入 | src_scan | masked_select;masked_select.out | masked_select;masked_select.out | False | exec_aclnn_masked_select;masked_select;masked_select_out | shared_by_2_ops;src_only |
| aclnnMaskedSoftmaxWithRelPosBias | 已接入 | yaml_exec | npu_masked_softmax_with_rel_pos_bias |  | True |  | yaml_only |
| aclnnMatmul | 已接入 | src_scan | npu_attn_softmax_backward_ |  | True | _exec_fft;matmul_implement_npu;npu_attn_softmax_backward_ | src_only |
| aclnnMatmulAllReduce | 已接入 | src_scan | npu_mm_all_reduce_base |  | True | npu_mm_all_reduce_base | src_only |
| aclnnMatmulAllReduceAddRmsNorm | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMatmulAllReduceV2 | 已接入 | src_scan | npu_mm_all_reduce_base |  | True | npu_mm_all_reduce_base | src_only |
| aclnnMatmulCompress | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMatmulCompressDequant | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMatmulReduceScatter | 已接入 | src_scan |  |  | True | size | src_only;src_hit_but_op_name_unresolved |
| aclnnMatmulReduceScatterV2 | 已接入 | src_scan | npu_quant_mm_reduce_scatter |  | True | npu_quant_mm_reduce_scatter;size | src_only |
| aclnnMatmulWeightNz | 已接入 | src_scan | mm;mm.out |  | True | matmul_implement_npu;mm;mm_out | shared_by_2_ops;src_only |
| aclnnMax | 已接入 | src_scan | max;max.dim;max.dim_max;max.names_dim;max.names_dim_max;max.out | max;max.dim;max.dim_max;max.names_dim;max.names_dim_max;max.out | False | max | shared_by_6_ops;src_only |
| aclnnMaxDim | 已接入 | src_scan | max;max.dim;max.dim_max;max.names_dim;max.names_dim_max;max.out |  | True | max;max_out | shared_by_6_ops;src_only |
| aclnnMaxN | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMaxPool | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMaxPool2dWithIndices | 已接入 | src_scan | max_pool2d_with_indices;max_pool2d_with_indices.out | max_pool2d_with_indices;max_pool2d_with_indices.out | False | exec_max_pool2d_with_indices;max_pool2d_with_indices;max_pool2d_with_indices_out | shared_by_2_ops;src_only |
| aclnnMaxPool2dWithIndicesBackward | 已接入 | src_scan | max_pool2d_with_indices_backward;max_pool2d_with_indices_backward.grad_input | max_pool2d_with_indices_backward;max_pool2d_with_indices_backward.grad_input | False | max_pool2d_with_indices_backward;max_pool2d_with_indices_backward_out | shared_by_2_ops;src_only |
| aclnnMaxPool2dWithMask | 已接入 | src_scan | max_pool2d_with_indices;max_pool2d_with_indices.out |  | True | exec_max_pool2d_with_indices;max_pool2d_with_indices;max_pool2d_with_indices_out | shared_by_2_ops;src_only |
| aclnnMaxPool2dWithMaskBackward | 已接入 | src_scan | max_pool2d_with_indices_backward;max_pool2d_with_indices_backward.grad_input |  | True | max_pool2d_with_indices_backward;max_pool2d_with_indices_backward_out | shared_by_2_ops;src_only |
| aclnnMaxPool3dWithArgmax | 已接入 | src_scan | max_pool3d_with_indices;max_pool3d_with_indices.out |  | True | exec_max_pool3d_with_indices;max_pool3d_with_indices;max_pool3d_with_indices_out | shared_by_2_ops;src_only |
| aclnnMaxPool3dWithArgmaxBackward | 已接入 | src_scan | max_pool3d_with_indices_backward;max_pool3d_with_indices_backward.grad_input |  | True | max_pool3d_with_indices_backward;max_pool3d_with_indices_backward_out | shared_by_2_ops;src_only |
| aclnnMaxUnpool2d | 已接入 | yaml_exec | max_unpool2d;max_unpool2d.out | max_unpool2d;max_unpool2d.out | False |  | shared_by_2_ops;yaml_only |
| aclnnMaxUnpool2dBackward | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMaxUnpool3d | 已接入 | yaml_exec | max_unpool3d;max_unpool3d.out | max_unpool3d;max_unpool3d.out | False |  | shared_by_2_ops;yaml_only |
| aclnnMaxUnpool3dBackward | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMaxV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMaximum | 已接入 | src_scan | max.out;maximum;maximum.out | maximum;maximum.out | False | check_tensor;max_out;maximum | shared_by_3_ops;src_only |
| aclnnMean | 已接入 | src_scan | mean;mean.dim;mean.names_dim;mean.names_out;mean.out | mean;mean.dim;mean.names_dim;mean.names_out;mean.out | False | mean;mean_out;scalar_type | shared_by_5_ops;src_only |
| aclnnMeanV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMedian | 已接入 | yaml_exec | median | median | False |  | yaml_only |
| aclnnMedianDim | 已接入 | yaml_exec | median.dim;median.dim_values |  | True |  | shared_by_2_ops;yaml_only |
| aclnnMin | 已接入 | src_scan | min;min.dim;min.dim_min;min.names_dim;min.names_dim_min;min.out | min;min.dim;min.dim_min;min.names_dim;min.names_dim_min;min.out | False | min | shared_by_6_ops;src_only |
| aclnnMinDim | 已接入 | src_scan | min;min.dim;min.dim_min;min.names_dim;min.names_dim_min;min.out |  | True | min;min_out | shared_by_6_ops;src_only |
| aclnnMinN | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMinimum | 已接入 | src_scan | min.out;minimum.out | minimum.out | False | min_out;minimum_out | shared_by_2_ops;src_only |
| aclnnMish | 已接入 | src_scan | mish;mish.out | mish;mish.out | False | mish;mish_out | shared_by_2_ops;src_only |
| aclnnMishBackward | 已接入 | yaml_exec | mish_backward | mish_backward | False |  | yaml_only |
| aclnnMlaPreprocess | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMlaPreprocessV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMlaProlog | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMlaPrologV2WeightNz | 已接入 | src_scan | npu_mla_prolog_v2 |  | True | npu_mla_prolog_v2 | src_only |
| aclnnMlaPrologV3WeightNz | 已接入 | src_scan | npu_mla_prolog_v3;npu_mla_prolog_v3_functional |  | True | npu_mla_prolog_v3;npu_mla_prolog_v3_functional | shared_by_2_ops;src_only |
| aclnnMm | 已接入 | src_scan | mm;mm.out;npu_linear;npu_linear_backward | mm;mm.out | False | mm;mm_out;npu_linear;npu_linear_backward | shared_by_4_ops;src_only |
| aclnnModulate | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnModulateBackward | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMoeComputeExpertTokens | 已接入 | yaml_exec | npu_moe_compute_expert_tokens |  | True |  | yaml_only |
| aclnnMoeDistributeCombine | 已接入 | src_scan | npu_moe_distribute_combine |  | True | npu_moe_distribute_combine | src_only |
| aclnnMoeDistributeCombineAddRmsNorm | 已接入 | src_scan | npu_moe_distribute_combine_add_rms_norm |  | True | npu_moe_distribute_combine_add_rms_norm | src_only |
| aclnnMoeDistributeCombineAddRmsNormV2 | 已接入 | src_scan | npu_moe_distribute_combine_add_rms_norm |  | True | npu_moe_distribute_combine_add_rms_norm | src_only |
| aclnnMoeDistributeCombineV2 | 已接入 | src_scan | npu_moe_distribute_combine_v2 |  | True | npu_moe_distribute_combine_v2 | src_only |
| aclnnMoeDistributeCombineV3 | 已接入 | src_scan | npu_moe_distribute_combine_v2 |  | True | npu_moe_distribute_combine_v2 | src_only |
| aclnnMoeDistributeCombineV4 | 已接入 | src_scan | npu_moe_distribute_combine_v2 |  | True | npu_moe_distribute_combine_v2 | src_only |
| aclnnMoeDistributeDispatch | 已接入 | src_scan | npu_moe_distribute_dispatch |  | True | npu_moe_distribute_dispatch | src_only |
| aclnnMoeDistributeDispatchV2 | 已接入 | src_scan | npu_moe_distribute_dispatch_v2 |  | True | npu_moe_distribute_dispatch_v2 | src_only |
| aclnnMoeDistributeDispatchV3 | 已接入 | src_scan | npu_moe_distribute_dispatch_v2 |  | True | npu_moe_distribute_dispatch_v2 | src_only |
| aclnnMoeDistributeDispatchV4 | 已接入 | src_scan | npu_moe_distribute_dispatch_v2 |  | True | npu_moe_distribute_dispatch_v2 | src_only |
| aclnnMoeFinalizeRouting | 已接入 | src_scan | npu_moe_finalize_routing |  | True | npu_moe_finalize_routing | src_only |
| aclnnMoeFinalizeRoutingV2 | 已接入 | src_scan | npu_moe_finalize_routing |  | True | npu_moe_finalize_routing | src_only |
| aclnnMoeFinalizeRoutingV2Grad | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMoeFusedTopk | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMoeGatingTopK | 已接入 | src_scan | npu_moe_gating_top_k |  | True | npu_moe_gating_top_k | src_only |
| aclnnMoeGatingTopKSoftmax | 已接入 | src_scan | npu_moe_gating_top_k_softmax |  | True | npu_moe_gating_top_k_softmax | src_only |
| aclnnMoeGatingTopKSoftmaxV2 | 已接入 | src_scan | npu_moe_gating_top_k_softmax_v2 |  | True | npu_moe_gating_top_k_softmax_v2 | src_only |
| aclnnMoeInitRouting | 已接入 | src_scan | npu_moe_init_routing |  | True | npu_moe_init_routing | src_only |
| aclnnMoeInitRoutingQuant | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMoeInitRoutingQuantV2 | 已接入 | src_scan | npu_moe_init_routing_quant |  | True | npu_moe_init_routing_quant | src_only |
| aclnnMoeInitRoutingV2 | 已接入 | src_scan | npu_moe_init_routing_v2 |  | True | npu_moe_init_routing_v2 | src_only |
| aclnnMoeInitRoutingV2Grad | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMoeInitRoutingV3 | 已接入 | src_scan | npu_moe_init_routing_v2 |  | True | npu_moe_init_routing_v2 | src_only |
| aclnnMoeTokenPermute | 已接入 | yaml_exec | npu_moe_token_permute |  | True |  | yaml_only |
| aclnnMoeTokenPermuteGrad | 已接入 | yaml_exec | npu_moe_token_permute_grad |  | True |  | yaml_only |
| aclnnMoeTokenPermuteWithEp | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMoeTokenPermuteWithEpGrad | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMoeTokenPermuteWithRoutingMap | 已接入 | src_scan | npu_moe_token_permute_with_routing_map |  | True | npu_moe_token_permute_with_routing_map | src_only |
| aclnnMoeTokenPermuteWithRoutingMapGrad | 已接入 | src_scan |  |  | True | npu_moe_token_permute_with_routing_map_grad_symint | src_only;src_hit_but_op_name_unresolved |
| aclnnMoeTokenUnpermute | 已接入 | yaml_exec | npu_moe_token_unpermute |  | True |  | yaml_only |
| aclnnMoeTokenUnpermuteGrad | 已接入 | yaml_exec | npu_moe_token_unpermute_grad |  | True |  | yaml_only |
| aclnnMoeTokenUnpermuteWithEp | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMoeTokenUnpermuteWithEpGrad | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMoeTokenUnpermuteWithRoutingMap | 已接入 | src_scan | _npu_moe_token_unpermute_with_routing_map |  | True | _npu_moe_token_unpermute_with_routing_map | src_only |
| aclnnMoeTokenUnpermuteWithRoutingMapGrad | 已接入 | src_scan | npu_moe_token_unpermute_with_routing_map_grad |  | True | npu_moe_token_unpermute_with_routing_map_grad | src_only |
| aclnnMoeUpdateExpert | 已接入 | yaml_exec | npu_moe_update_expert |  | True |  | yaml_only |
| aclnnMrgbaCustom | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMseLoss | 已接入 | src_scan | mse_loss;mse_loss.out | mse_loss;mse_loss.out | False | mse_loss;scalar_type | shared_by_2_ops;src_only |
| aclnnMseLossBackward | 已接入 | yaml_exec | mse_loss_backward;mse_loss_backward.grad_input | mse_loss_backward;mse_loss_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnMseLossOut | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMul | 已接入 | src_scan | mul.Scalar;mul.Tensor;mul.out | mul.Scalar;mul.Tensor;mul.out | False | mul;mul_out;mul_out_npu_no_check | shared_by_3_ops;src_only |
| aclnnMuls | 已接入 | src_scan | mul.Scalar;mul.Tensor;mul.out |  | True | mul;mul_out;mul_out_npu_no_check | shared_by_3_ops;src_only |
| aclnnMultiScaleDeformableAttentionGrad | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMultiScaleDeformableAttnFunction | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnMultilabelMarginLoss | 已接入 | src_scan | multilabel_margin_loss.out;multilabel_margin_loss_forward;multilabel_margin_loss_forward.output | multilabel_margin_loss.out | False | multilabel_margin_loss_forward;multilabel_margin_loss_forward_out;multilabel_margin_loss_out | shared_by_3_ops;src_only |
| aclnnMultinomial | 已接入 | src_scan | multinomial;multinomial.out | multinomial;multinomial.out | False | currentStreamCaptureStatusMayInitCtx;multinomial;multinomial_out;multinomial_top_k_top_p_sample;multinomial_top_k_top_p_sample_op_api | shared_by_2_ops;src_only |
| aclnnMultinomialTensor | 已接入 | src_scan |  |  | True | currentStreamCaptureStatusMayInitCtx;multinomial_top_k_top_p_sample_op_api | src_only;src_hit_but_op_name_unresolved |
| aclnnMv | 已接入 | src_scan | mv;mv.out | mv;mv.out | False | mv;mv_out | shared_by_2_ops;src_only |
| aclnnNLLLoss | 已接入 | src_scan | nll_loss_forward;nll_loss_forward.output |  | True | nll_loss_forward;nll_loss_forward_out | shared_by_2_ops;src_only |
| aclnnNLLLoss2d | 已接入 | src_scan | nll_loss2d_forward;nll_loss2d_forward.output |  | True | Tensor;nll_loss2d_forward | shared_by_2_ops;src_only |
| aclnnNLLLoss2dBackward | 已接入 | src_scan | nll_loss2d_backward;nll_loss2d_backward.grad_input |  | True | Tensor;nll_loss2d_backward | shared_by_2_ops;src_only |
| aclnnNLLLossBackward | 已接入 | src_scan | nll_loss_backward;nll_loss_backward.grad_input |  | True | Tensor;nll_loss_backward | shared_by_2_ops;src_only |
| aclnnNanMedian | 已接入 | yaml_exec | nanmedian |  | True |  | yaml_only |
| aclnnNanMedianDim | 已接入 | yaml_exec | nanmedian.dim |  | True |  | yaml_only |
| aclnnNanToNum | 已接入 | src_scan | nan_to_num;nan_to_num.out | nan_to_num;nan_to_num.out | False | nan_to_num;nan_to_num_out | shared_by_2_ops;src_only |
| aclnnNeScalar | 已接入 | src_scan | ne.Scalar;ne.Scalar_out;ne.Tensor;ne.Tensor_out |  | True | ne;ne_out | shared_by_4_ops;src_only |
| aclnnNeTensor | 已接入 | src_scan | ne.Scalar;ne.Scalar_out;ne.Tensor;ne.Tensor_out |  | True | ne;ne_out | shared_by_4_ops;src_only |
| aclnnNeg | 已接入 | yaml_exec | neg;neg.out;neg_ | neg;neg.out | False |  | shared_by_3_ops;yaml_only |
| aclnnNonMaxSuppression | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnNonzero | 已接入 | src_scan | nonzero;nonzero.out | nonzero;nonzero.out | False | nonzero;nonzero_out | shared_by_2_ops;src_only |
| aclnnNonzeroV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnNorm | 已接入 | src_scan | norm.Scalar;norm.ScalarOpt_dim;norm.ScalarOpt_dim_dtype;norm.ScalarOpt_dtype;norm.dtype_out;norm.out | norm.Scalar;norm.ScalarOpt_dim;norm.ScalarOpt_dim_dtype;norm.ScalarOpt_dtype;norm.dtype_out;norm.out | False | norm;norm_out;norm_out_npu_nocheck_opapi | shared_by_6_ops;src_only |
| aclnnNormRopeConcat | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnNormRopeConcatBackward | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnNormalFloatFloat | 已接入 | src_scan | normal.Tensor_Tensor;normal.Tensor_Tensor_out;normal.Tensor_float;normal.Tensor_float_out;normal.float_Tensor;normal.float_Tensor_out;normal.float_float;normal.float_float_out |  | True | normal;normal_out | shared_by_8_ops;src_only |
| aclnnNormalFloatTensor | 已接入 | src_scan | normal.Tensor_Tensor;normal.Tensor_Tensor_out;normal.Tensor_float;normal.Tensor_float_out;normal.float_Tensor;normal.float_Tensor_out;normal.float_float;normal.float_float_out |  | True | normal;normal_out | shared_by_8_ops;src_only |
| aclnnNormalTensorFloat | 已接入 | src_scan | normal.Tensor_Tensor;normal.Tensor_Tensor_out;normal.Tensor_float;normal.Tensor_float_out;normal.float_Tensor;normal.float_Tensor_out;normal.float_float;normal.float_float_out |  | True | normal;normal_out | shared_by_8_ops;src_only |
| aclnnNormalTensorTensor | 已接入 | src_scan | normal.Tensor_Tensor;normal.Tensor_Tensor_out;normal.Tensor_float;normal.Tensor_float_out;normal.float_Tensor;normal.float_Tensor_out;normal.float_float;normal.float_float_out |  | True | normal;normal_out | shared_by_8_ops;src_only |
| aclnnNpuFormatCast | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnNsaCompress | 已接入 | yaml_exec | npu_nsa_compress |  | True |  | yaml_only |
| aclnnNsaCompressAttention | 已接入 | yaml_exec | npu_nsa_compress_attention |  | True |  | yaml_only |
| aclnnNsaCompressAttentionInfer | 已接入 | yaml_exec | npu_nsa_compress_attention_infer |  | True |  | yaml_only |
| aclnnNsaCompressGrad | 已接入 | yaml_exec | npu_nsa_compress_grad |  | True |  | yaml_only |
| aclnnNsaCompressWithCache | 已接入 | yaml_exec | npu_nsa_compress_infer.cache |  | True |  | yaml_only |
| aclnnNsaSelectedAttention | 已接入 | yaml_exec | npu_nsa_select_attention |  | True |  | yaml_only |
| aclnnNsaSelectedAttentionGrad | 已接入 | yaml_exec | npu_nsa_select_attention_grad |  | True |  | yaml_only |
| aclnnNsaSelectedAttentionInfer | 已接入 | yaml_exec | npu_nsa_select_attention_infer |  | True |  | yaml_only |
| aclnnObfuscationCalculate | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnObfuscationCalculateV2 | 已接入 | src_scan | obfuscation_calculate |  | True | obfuscation_calculate | src_only |
| aclnnObfuscationSetup | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnObfuscationSetupV2 | 已接入 | src_scan | obfuscation_finalize;obfuscation_initialize |  | True | obfuscation_finalize;obfuscation_initialize | shared_by_2_ops;src_only |
| aclnnOneHot | 已接入 | src_scan | npu_one_hot;one_hot | one_hot | False | npu_one_hot;one_hot | shared_by_2_ops;src_only |
| aclnnPdist | 已接入 | src_scan | _pdist_forward |  | True | _pdist_forward | src_only |
| aclnnPdistForward | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnPermute | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnPolar | 已接入 | yaml_exec | polar;polar.out | polar;polar.out | False |  | shared_by_2_ops;yaml_only |
| aclnnPowScalarTensor | 已接入 | yaml_exec | pow.Scalar;pow.Scalar_out |  | True |  | shared_by_2_ops;yaml_only |
| aclnnPowTensorScalar | 已接入 | yaml_exec | pow.Tensor_Scalar;pow.Tensor_Scalar_out |  | True |  | shared_by_2_ops;yaml_only |
| aclnnPowTensorTensor | 已接入 | yaml_exec | pow.Tensor_Tensor;pow.Tensor_Tensor_out |  | True |  | shared_by_2_ops;yaml_only |
| aclnnPrecisionCompare | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnPrelu | 已接入 | yaml_exec | _prelu_kernel |  | True |  | yaml_only |
| aclnnPreluBackward | 已接入 | src_scan | _prelu_kernel_backward |  | True | _prelu_kernel_backward | src_only |
| aclnnProd | 已接入 | src_scan | prod;prod.dim_int;prod.int_out | prod;prod.dim_int;prod.int_out | False | prod | shared_by_3_ops;src_only |
| aclnnProdDim | 已接入 | src_scan | prod;prod.dim_int;prod.int_out |  | True | check_tensor;prod | shared_by_3_ops;src_only |
| aclnnPromptFlashAttention | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnPromptFlashAttentionV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnPromptFlashAttentionV3 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnQr | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnQuantConvolution | 已接入 | src_scan | npu_quant_conv2d |  | True | npu_quant_conv2d;npu_quant_conv2d_out | src_only |
| aclnnQuantGroupedMatmulDequant | 已接入 | yaml_exec | npu_quant_grouped_matmul_dequant |  | True |  | yaml_only |
| aclnnQuantMatmul | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnQuantMatmulAllReduce | 已接入 | src_scan | npu_mm_all_reduce_base |  | True | npu_mm_all_reduce_base | src_only |
| aclnnQuantMatmulAllReduceAddRmsNorm | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnQuantMatmulAllReduceV2 | 已接入 | src_scan | npu_mm_all_reduce_base |  | True | npu_mm_all_reduce_base | src_only |
| aclnnQuantMatmulAllReduceV3 | 已接入 | src_scan | npu_mm_all_reduce_base |  | True | npu_mm_all_reduce_base | src_only |
| aclnnQuantMatmulDequant | 已接入 | yaml_exec | npu_quant_matmul_dequant |  | True |  | yaml_only |
| aclnnQuantMatmulReduceSumWeightNz | 已接入 | src_scan | npu_quant_matmul_reduce_sum |  | True | npu_quant_matmul_reduce_sum | src_only |
| aclnnQuantMatmulV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnQuantMatmulV3 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnQuantMatmulV4 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnQuantMatmulV5 | 已接入 | src_scan | npu_quant_matmul |  | True | npu_quant_matmul | src_only |
| aclnnQuantMatmulWeightNz | 已接入 | src_scan | npu_quant_matmul |  | True | npu_quant_matmul | src_only |
| aclnnQuantize | 已接入 | src_scan | _quantize_per_channel_impl.out;_quantize_per_tensor_impl.out |  | True | _quantize_per_channel_impl_out;_quantize_per_tensor_impl_out;npu_quantize_by_kernel | shared_by_2_ops;src_only |
| aclnnQuantizedBatchNorm | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnRReluWithNoise | 已接入 | src_scan | rrelu_with_noise;rrelu_with_noise.out |  | True | rrelu_with_noise;rrelu_with_noise_out | shared_by_2_ops;src_only |
| aclnnRainFusionAttention | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnRandperm | 已接入 | src_scan | randperm;randperm.generator;randperm.generator_out;randperm.out | randperm;randperm.generator;randperm.generator_out;randperm.out | False | randperm;randperm_out | shared_by_4_ops;src_only |
| aclnnRange | 已接入 | src_scan | range;range.out;range.step | range;range.out;range.step | False | range;range_out | shared_by_3_ops;src_only |
| aclnnReal | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnReciprocal | 已接入 | yaml_exec | reciprocal;reciprocal.out | reciprocal;reciprocal.out | False |  | shared_by_2_ops;yaml_only |
| aclnnRecurrentGatedDeltaRule | 已接入 | src_scan | npu_recurrent_gated_delta_rule;npu_recurrent_gated_delta_rule_functional |  | True | npu_recurrent_gated_delta_rule;npu_recurrent_gated_delta_rule_functional | shared_by_2_ops;src_only |
| aclnnReduceLogSum | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnReduceNansum | 已接入 | src_scan |  |  | True | scalar_type | src_only;src_hit_but_op_name_unresolved |
| aclnnReduceSum | 已接入 | src_scan | sum;sum.DimnameList_out;sum.IntList_out;sum.dim_DimnameList;sum.dim_IntList |  | True | sum | shared_by_5_ops;src_only |
| aclnnReflectionPad1d | 已接入 | yaml_exec | reflection_pad1d;reflection_pad1d.out | reflection_pad1d;reflection_pad1d.out | False |  | shared_by_2_ops;yaml_only |
| aclnnReflectionPad1dBackward | 已接入 | yaml_exec | reflection_pad1d_backward;reflection_pad1d_backward.grad_input | reflection_pad1d_backward;reflection_pad1d_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnReflectionPad2d | 已接入 | yaml_exec | reflection_pad2d;reflection_pad2d.out | reflection_pad2d;reflection_pad2d.out | False |  | shared_by_2_ops;yaml_only |
| aclnnReflectionPad2dBackward | 已接入 | yaml_exec | reflection_pad2d_backward;reflection_pad2d_backward.grad_input | reflection_pad2d_backward;reflection_pad2d_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnReflectionPad3d | 已接入 | yaml_exec | reflection_pad3d;reflection_pad3d.out | reflection_pad3d;reflection_pad3d.out | False |  | shared_by_2_ops;yaml_only |
| aclnnReflectionPad3dBackward | 已接入 | yaml_exec | reflection_pad3d_backward;reflection_pad3d_backward.grad_input | reflection_pad3d_backward;reflection_pad3d_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnRelu | 已接入 | yaml_exec | relu | relu | False |  | yaml_only |
| aclnnRemainderScalarTensor | 已接入 | src_scan | remainder.Scalar;remainder.Scalar_Tensor;remainder.Scalar_out;remainder.Tensor;remainder.Tensor_out |  | True | remainder | shared_by_5_ops;src_only |
| aclnnRemainderTensorScalar | 已接入 | src_scan | remainder.Scalar;remainder.Scalar_Tensor;remainder.Scalar_out;remainder.Tensor;remainder.Tensor_out |  | True | remainder;remainder_out | shared_by_5_ops;src_only |
| aclnnRemainderTensorTensor | 已接入 | src_scan | remainder.Scalar;remainder.Scalar_Tensor;remainder.Scalar_out;remainder.Tensor;remainder.Tensor_out |  | True | remainder;remainder_out | shared_by_5_ops;src_only |
| aclnnRenorm | 已接入 | src_scan | renorm;renorm.out | renorm;renorm.out | False | renorm;renorm_out | shared_by_2_ops;src_only |
| aclnnRepeat | 已接入 | yaml_exec | repeat | repeat | False |  | yaml_only |
| aclnnRepeatInterleave | 已接入 | src_scan | repeat_interleave.self_Tensor;repeat_interleave.self_int | repeat_interleave.self_Tensor;repeat_interleave.self_int | False | repeat_interleave;repeat_interleave_symint | shared_by_2_ops;src_only |
| aclnnRepeatInterleaveInt | 已接入 | src_scan |  |  | True | repeat_interleave_symint | src_only;src_hit_but_op_name_unresolved |
| aclnnRepeatInterleaveIntWithDim | 已接入 | src_scan |  |  | True | repeat_interleave_symint | src_only;src_hit_but_op_name_unresolved |
| aclnnRepeatInterleaveTensor | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnRepeatInterleaveWithDim | 已接入 | src_scan | repeat_interleave.self_Tensor;repeat_interleave.self_int |  | True | repeat_interleave;repeat_interleave_symint | shared_by_2_ops;src_only |
| aclnnReplicationPad1d | 已接入 | yaml_exec | replication_pad1d;replication_pad1d.out | replication_pad1d;replication_pad1d.out | False |  | shared_by_2_ops;yaml_only |
| aclnnReplicationPad1dBackward | 已接入 | yaml_exec | replication_pad1d_backward;replication_pad1d_backward.grad_input | replication_pad1d_backward;replication_pad1d_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnReplicationPad2d | 已接入 | yaml_exec | replication_pad2d;replication_pad2d.out | replication_pad2d;replication_pad2d.out | False |  | shared_by_2_ops;yaml_only |
| aclnnReplicationPad2dBackward | 已接入 | yaml_exec | replication_pad2d_backward;replication_pad2d_backward.grad_input | replication_pad2d_backward;replication_pad2d_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnReplicationPad3d | 已接入 | yaml_exec | replication_pad3d;replication_pad3d.out | replication_pad3d;replication_pad3d.out | False |  | shared_by_2_ops;yaml_only |
| aclnnReplicationPad3dBackward | 已接入 | yaml_exec | replication_pad3d_backward;replication_pad3d_backward.grad_input | replication_pad3d_backward;replication_pad3d_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnResize | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnRightShift | 已接入 | src_scan | __irshift__.Scalar;__irshift__.Tensor;__rshift__.Scalar;__rshift__.Tensor |  | True | __irshift__;__rshift__ | shared_by_4_ops;src_only |
| aclnnRingAttentionUpdate | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnRingAttentionUpdateV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnRmsNorm | 已接入 | src_scan | npu_rms_norm |  | True | npu_rms_norm | src_only |
| aclnnRmsNormGrad | 已接入 | src_scan | npu_rms_norm_backward |  | True | npu_rms_norm_backward | src_only |
| aclnnRmsNormQuant | 已接入 | src_scan | npu_rms_norm_quant |  | True | npu_rms_norm_quant | src_only |
| aclnnRoiAlign | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnRoiAlignV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnRoiAlignV2Backward | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnRoll | 已接入 | yaml_exec | roll | roll | False |  | yaml_only |
| aclnnRopeWithSinCosCache | 已接入 | yaml_exec | npu_mrope |  | True |  | yaml_only |
| aclnnRotaryPositionEmbedding | 已接入 | src_scan | npu_rotary_mul |  | True | npu_rotary_mul | src_only |
| aclnnRotaryPositionEmbeddingGrad | 已接入 | src_scan | npu_rotary_mul_backward |  | True | npu_rotary_mul_backward | src_only |
| aclnnRound | 已接入 | yaml_exec | round;round.out | round;round.out | False |  | shared_by_2_ops;yaml_only |
| aclnnRoundDecimals | 已接入 | src_scan | round;round.decimals;round.decimals_out;round.out |  | True | round;round_out | shared_by_4_ops;src_only |
| aclnnRsqrt | 已接入 | yaml_exec | rsqrt;rsqrt.out | rsqrt;rsqrt.out | False |  | shared_by_2_ops;yaml_only |
| aclnnRsub | 已接入 | yaml_exec | rsub.Tensor | rsub.Tensor | False |  | yaml_only |
| aclnnRsubs | 已接入 | yaml_exec | rsub.Scalar |  | True |  | yaml_only |
| aclnnSWhere | 已接入 | src_scan | where;where.self;where.self_out |  | True | where;where_out | shared_by_3_ops;src_only |
| aclnnScale | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnScaledMaskedSoftmax | 已接入 | src_scan | _masked_softmax;npu_scaled_masked_softmax |  | True | _masked_softmax;npu_scaled_masked_softmax | shared_by_2_ops;src_only |
| aclnnScaledMaskedSoftmaxBackward | 已接入 | src_scan | npu_scaled_masked_softmax_backward |  | True | npu_scaled_masked_softmax_backward | src_only |
| aclnnScatter | 已接入 | src_scan | scatter.src_out;scatter.value_out | scatter.src_out;scatter.value_out | False | scatter_out | shared_by_2_ops;src_only |
| aclnnScatterAdd | 已接入 | src_scan | scatter_add;scatter_add.dimname;scatter_add_ | scatter_add;scatter_add.dimname | False | scatter_add;scatter_add_ | shared_by_3_ops;src_only |
| aclnnScatterNd | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnScatterNdUpdate | 已接入 | src_scan | npu_scatter_nd_update;npu_scatter_nd_update_ |  | True | npu_scatter_nd_update;npu_scatter_nd_update_ | shared_by_2_ops;src_only |
| aclnnScatterPaKvCache | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnScatterValue | 已接入 | src_scan | scatter.src_out;scatter.value_out |  | True | scatter_out | shared_by_2_ops;src_only |
| aclnnSearchSorted | 已接入 | yaml_exec | searchsorted.Tensor;searchsorted.Tensor_out |  | True |  | shared_by_2_ops;yaml_only |
| aclnnSearchSorteds | 已接入 | yaml_exec | searchsorted.Scalar |  | True |  | yaml_only |
| aclnnSelu | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSeluBackward | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnShrink | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSigmoid | 已接入 | yaml_exec | sigmoid;sigmoid.out | sigmoid;sigmoid.out | False |  | shared_by_2_ops;yaml_only |
| aclnnSigmoidBackward | 已接入 | yaml_exec | sigmoid_backward;sigmoid_backward.grad_input | sigmoid_backward;sigmoid_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnSign | 已接入 | yaml_exec+src_scan | sgn;sgn.out;sign;sign.out;sign_ | sign;sign.out | False | sgn;sgn_out | shared_by_5_ops;yaml+src |
| aclnnSignBitsPack | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSignBitsUnpack | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSignbit | 已接入 | yaml_exec | signbit.out | signbit.out | False |  | yaml_only |
| aclnnSilentCheck | 已接入 | src_scan | _npu_silent_check_v2 |  | True | _npu_silent_check_v2 | src_only |
| aclnnSilentCheckV2 | 已接入 | src_scan | _npu_silent_check_v3 |  | True | _npu_silent_check_v3 | src_only |
| aclnnSilu | 已接入 | yaml_exec | silu;silu.out;silu_ | silu;silu.out | False |  | shared_by_3_ops;yaml_only |
| aclnnSiluBackward | 已接入 | src_scan | silu_backward;silu_backward.grad_input | silu_backward;silu_backward.grad_input | False | silu_backward | shared_by_2_ops;src_only |
| aclnnSimThreadExponential | 已接入 | src_scan | npu_sim_exponential_ |  | True | npu_sim_exponential_ | src_only |
| aclnnSin | 已接入 | yaml_exec | sin;sin.out | sin;sin.out | False |  | shared_by_2_ops;yaml_only |
| aclnnSinc | 已接入 | yaml_exec | sinc;sinc.out | sinc;sinc.out | False |  | shared_by_2_ops;yaml_only |
| aclnnSinh | 已接入 | yaml_exec | sinh;sinh.out | sinh;sinh.out | False |  | shared_by_2_ops;yaml_only |
| aclnnSinkhorn | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSlice | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSliceV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSlogdet | 已接入 | src_scan | slogdet | slogdet | False | slogdet | src_only |
| aclnnSmoothL1Loss | 已接入 | src_scan | smooth_l1_loss;smooth_l1_loss.out | smooth_l1_loss;smooth_l1_loss.out | False | smooth_l1_loss;smooth_l1_loss_out | shared_by_2_ops;src_only |
| aclnnSmoothL1LossBackward | 已接入 | src_scan | smooth_l1_loss_backward;smooth_l1_loss_backward.grad_input | smooth_l1_loss_backward;smooth_l1_loss_backward.grad_input | False | smooth_l1_loss_backward;smooth_l1_loss_backward_out | shared_by_2_ops;src_only |
| aclnnSoftMarginLoss | 已接入 | yaml_exec | soft_margin_loss;soft_margin_loss.out | soft_margin_loss;soft_margin_loss.out | False |  | shared_by_2_ops;yaml_only |
| aclnnSoftMarginLossBackward | 已接入 | yaml_exec | soft_margin_loss_backward;soft_margin_loss_backward.grad_input | soft_margin_loss_backward;soft_margin_loss_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnSoftmax | 已接入 | yaml_exec | _softmax;_softmax.out;npu_attn_softmax_ | _softmax;_softmax.out | False |  | shared_by_3_ops;yaml_only |
| aclnnSoftmaxBackward | 已接入 | yaml_exec+src_scan | _softmax_backward_data;_softmax_backward_data.out;npu_attn_softmax_backward_ |  | True | npu_attn_softmax_backward_ | shared_by_3_ops;yaml+src |
| aclnnSoftmaxCrossEntropyWithLogits | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSoftplus | 已接入 | yaml_exec | softplus;softplus.out | softplus;softplus.out | False |  | shared_by_2_ops;yaml_only |
| aclnnSoftplusBackward | 已接入 | yaml_exec | softplus_backward.grad_input | softplus_backward.grad_input | False |  | yaml_only |
| aclnnSoftshrink | 已接入 | yaml_exec | softshrink;softshrink.out | softshrink;softshrink.out | False |  | shared_by_2_ops;yaml_only |
| aclnnSoftshrinkBackward | 已接入 | yaml_exec | softshrink_backward;softshrink_backward.grad_input | softshrink_backward;softshrink_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnSort | 已接入 | src_scan | sort;sort.dimname;sort.dimname_values;sort.stable;sort.values;sort.values_stable | sort;sort.dimname;sort.dimname_values;sort.stable;sort.values;sort.values_stable | False | sort;sort_out;sort_output | shared_by_6_ops;src_only |
| aclnnSparseFlashAttentionGrad | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSparseLightningIndexerGradKLLoss | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSplitTensor | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSplitWithSize | 已接入 | src_scan | split_with_sizes_copy.out |  | True | split_with_sizes_copy_out | src_only |
| aclnnSqrt | 已接入 | yaml_exec | sqrt;sqrt.out | sqrt;sqrt.out | False |  | shared_by_2_ops;yaml_only |
| aclnnSquaredRelu | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnStack | 已接入 | yaml_exec | stack;stack.out | stack;stack.out | False |  | shared_by_2_ops;yaml_only |
| aclnnStd | 已接入 | src_scan | std.correction;std.correction_out | std.correction;std.correction_out | False | std;std_out | shared_by_2_ops;src_only |
| aclnnStdMeanCorrection | 已接入 | src_scan | std_mean.correction |  | True | std_mean | src_only |
| aclnnStridedSliceAssignV2 | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSub | 已接入 | src_scan | sub.Scalar;sub.Tensor;sub.out | sub.Scalar;sub.Tensor;sub.out | False | sub;sub_out;sub_out_npu_nocheck | shared_by_3_ops;src_only |
| aclnnSubs | 已接入 | src_scan | sub.Scalar;sub.Tensor;sub.out |  | True | sub;sub_out;sub_out_npu_nocheck | shared_by_3_ops;src_only |
| aclnnSum | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSvd | 已接入 | yaml_exec | _linalg_svd.U |  | True |  | yaml_only |
| aclnnSwiGlu | 已接入 | yaml_exec | npu_swiglu |  | True |  | yaml_only |
| aclnnSwiGluGrad | 已接入 | yaml_exec | npu_swiglu_backward |  | True |  | yaml_only |
| aclnnSwiGluQuant | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSwiGluQuantV2 | 已接入 | src_scan | npu_swiglu_quant |  | True | npu_swiglu_quant | src_only |
| aclnnSwinAttentionScoreQuant | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSwinTransformerLnQkvQuant | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSwish | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSwishBackward | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnSyncBatchNormGatherStats | 已接入 | src_scan | batch_norm_gather_stats_update |  | True | batch_norm_gather_stats_update;batch_norm_gather_stats_update_npu_impl | src_only |
| aclnnTake | 已接入 | yaml_exec | take;take.out | take;take.out | False |  | shared_by_2_ops;yaml_only |
| aclnnTan | 已接入 | yaml_exec | tan;tan.out | tan;tan.out | False |  | shared_by_2_ops;yaml_only |
| aclnnTanh | 已接入 | src_scan | tanh;tanh.out | tanh;tanh.out | False | tanh;tanh_out | shared_by_2_ops;src_only |
| aclnnTanhBackward | 已接入 | yaml_exec | tanh_backward;tanh_backward.grad_input | tanh_backward;tanh_backward.grad_input | False |  | shared_by_2_ops;yaml_only |
| aclnnTfScatterAdd | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnThreeInterpolateBackward | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnThreshold | 已接入 | src_scan | threshold;threshold.out | threshold;threshold.out | False | threshold;threshold_out | shared_by_2_ops;src_only |
| aclnnThresholdBackward | 已接入 | yaml_exec | threshold_backward | threshold_backward | False |  | yaml_only |
| aclnnTopKTopPSample | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnTopk | 已接入 | src_scan | topk;topk.values | topk;topk.values | False | topk;topk_out | shared_by_2_ops;src_only |
| aclnnTrace | 已接入 | yaml_exec | trace | trace | False |  | yaml_only |
| aclnnTransConvolutionWeight | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnTransMatmulWeight | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnTransQuantParam | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnTransQuantParamV2 | 已接入 | src_scan | npu_trans_quant_param;npu_weight_quant_batchmatmul |  | True | npu_trans_quant_param;npu_weight_quant_batchmatmul | shared_by_2_ops;src_only |
| aclnnTransQuantParamV3 | 已接入 | src_scan | npu_trans_quant_param |  | True | npu_trans_quant_param | src_only |
| aclnnTransformBiasRescaleQkv | 已接入 | src_scan | _transform_bias_rescale_qkv | _transform_bias_rescale_qkv | False | _transform_bias_rescale_qkv | src_only |
| aclnnTransposeBatchMatMul | 已接入 | yaml_exec | npu_transpose_batchmatmul |  | True |  | yaml_only |
| aclnnTriangularSolve | 已接入 | yaml_exec+src_scan | triangular_solve.X | triangular_solve.X | False | exec_triangular_solve | yaml+src;src_hit_but_op_name_unresolved |
| aclnnTril | 已接入 | yaml_exec | tril;tril.out | tril;tril.out | False |  | shared_by_2_ops;yaml_only |
| aclnnTriu | 已接入 | yaml_exec | triu;triu.out | triu;triu.out | False |  | shared_by_2_ops;yaml_only |
| aclnnTrunc | 已接入 | yaml_exec | trunc;trunc.out | trunc;trunc.out | False |  | shared_by_2_ops;yaml_only |
| aclnnUnfoldGrad | 已接入 | src_scan | unfold_backward |  | True | unfold_backward | src_only |
| aclnnUnique | 已接入 | src_scan | _unique | _unique | False | _unique | src_only |
| aclnnUnique2 | 已接入 | src_scan | _unique2 | _unique2 | False | _unique2 | src_only |
| aclnnUniqueConsecutive | 已接入 | src_scan | unique_consecutive | unique_consecutive | False | unique_consecutive | src_only |
| aclnnUniqueDim | 已接入 | src_scan | unique_dim | unique_dim | False | unique_dim | src_only |
| aclnnUpsampleBicubic2d | 已接入 | src_scan | upsample_bicubic2d;upsample_bicubic2d.out | upsample_bicubic2d;upsample_bicubic2d.out | False | upsample_bicubic2d;upsample_bicubic2d_opapi;upsample_bicubic2d_out | shared_by_2_ops;src_only |
| aclnnUpsampleBicubic2dAA | 已接入 | src_scan | _upsample_bicubic2d_aa;_upsample_bicubic2d_aa.out |  | True | _upsample_bicubic2d_aa;_upsample_bicubic2d_aa_out | shared_by_2_ops;src_only |
| aclnnUpsampleBicubic2dAAGrad | 已接入 | src_scan | _upsample_bicubic2d_aa_backward;_upsample_bicubic2d_aa_backward.grad_input |  | True | _upsample_bicubic2d_aa_backward;_upsample_bicubic2d_aa_backward_out | shared_by_2_ops;src_only |
| aclnnUpsampleBicubic2dBackward | 已接入 | src_scan | upsample_bicubic2d_backward;upsample_bicubic2d_backward.grad_input | upsample_bicubic2d_backward;upsample_bicubic2d_backward.grad_input | False | upsample_bicubic2d_backward;upsample_bicubic2d_backward_out | shared_by_2_ops;src_only |
| aclnnUpsampleBilinear2d | 已接入 | src_scan | upsample_bilinear2d;upsample_bilinear2d.out | upsample_bilinear2d;upsample_bilinear2d.out | False | upsample_bilinear2d;upsample_bilinear2d_out | shared_by_2_ops;src_only |
| aclnnUpsampleBilinear2dAA | 已接入 | src_scan | _upsample_bilinear2d_aa;_upsample_bilinear2d_aa.out |  | True | _upsample_bilinear2d_aa;_upsample_bilinear2d_aa_out | shared_by_2_ops;src_only |
| aclnnUpsampleBilinear2dAABackward | 已接入 | src_scan | _upsample_bilinear2d_aa_backward;_upsample_bilinear2d_aa_backward.grad_input |  | True | _upsample_bilinear2d_aa_backward;_upsample_bilinear2d_aa_backward_out | shared_by_2_ops;src_only |
| aclnnUpsampleBilinear2dBackward | 已接入 | src_scan |  |  | True | upsample_bilinear2d_backward_old;upsample_bilinear2d_backward_old_out | src_only;src_hit_but_op_name_unresolved |
| aclnnUpsampleBilinear2dBackwardV2 | 已接入 | src_scan | upsample_bilinear2d_backward;upsample_bilinear2d_backward.grad_input |  | True | upsample_bilinear2d_backward;upsample_bilinear2d_backward_out | shared_by_2_ops;src_only |
| aclnnUpsampleLinear1d | 已接入 | src_scan | upsample_linear1d;upsample_linear1d.out | upsample_linear1d;upsample_linear1d.out | False | upsample_linear1d;upsample_linear1d_out | shared_by_2_ops;src_only |
| aclnnUpsampleLinear1dBackward | 已接入 | src_scan | upsample_linear1d_backward | upsample_linear1d_backward | False | upsample_linear1d_backward | src_only |
| aclnnUpsampleNearest1d | 已接入 | src_scan |  |  | True | upsample_nearest1d_old;upsample_nearest1d_old_out | src_only;src_hit_but_op_name_unresolved |
| aclnnUpsampleNearest1dBackward | 已接入 | src_scan | upsample_nearest1d_backward;upsample_nearest1d_backward.grad_input | upsample_nearest1d_backward;upsample_nearest1d_backward.grad_input | False | upsample_nearest1d_backward;upsample_nearest1d_backward_out | shared_by_2_ops;src_only |
| aclnnUpsampleNearest1dV2 | 已接入 | src_scan | upsample_nearest1d;upsample_nearest1d.out |  | True | upsample_nearest1d;upsample_nearest1d_out | shared_by_2_ops;src_only |
| aclnnUpsampleNearest2d | 已接入 | src_scan |  |  | True | upsample_nearest2d_old;upsample_nearest2d_old_out | src_only;src_hit_but_op_name_unresolved |
| aclnnUpsampleNearest2dBackward | 已接入 | src_scan | upsample_nearest2d_backward;upsample_nearest2d_backward.grad_input | upsample_nearest2d_backward;upsample_nearest2d_backward.grad_input | False | upsample_nearest2d_backward;upsample_nearest2d_backward_out | shared_by_2_ops;src_only |
| aclnnUpsampleNearest2dV2 | 已接入 | src_scan | upsample_nearest2d;upsample_nearest2d.out |  | True | upsample_nearest2d;upsample_nearest2d_out | shared_by_2_ops;src_only |
| aclnnUpsampleNearest3d | 已接入 | src_scan | upsample_nearest3d;upsample_nearest3d.out | upsample_nearest3d;upsample_nearest3d.out | False | upsample_nearest3d;upsample_nearest3d_out | shared_by_2_ops;src_only |
| aclnnUpsampleNearest3dBackward | 已接入 | src_scan | upsample_nearest3d_backward;upsample_nearest3d_backward.grad_input | upsample_nearest3d_backward;upsample_nearest3d_backward.grad_input | False | upsample_nearest3d_backward;upsample_nearest3d_backward_out | shared_by_2_ops;src_only |
| aclnnUpsampleNearestExact1d | 已接入 | src_scan | _upsample_nearest_exact1d;_upsample_nearest_exact1d.out | _upsample_nearest_exact1d;_upsample_nearest_exact1d.out | False | _upsample_nearest_exact1d;_upsample_nearest_exact1d_out | shared_by_2_ops;src_only |
| aclnnUpsampleNearestExact1dBackward | 已接入 | src_scan | _upsample_nearest_exact1d_backward;_upsample_nearest_exact1d_backward.grad_input | _upsample_nearest_exact1d_backward;_upsample_nearest_exact1d_backward.grad_input | False | _upsample_nearest_exact1d_backward;_upsample_nearest_exact1d_backward_out | shared_by_2_ops;src_only |
| aclnnUpsampleNearestExact2d | 已接入 | src_scan | _upsample_nearest_exact2d;_upsample_nearest_exact2d.out | _upsample_nearest_exact2d;_upsample_nearest_exact2d.out | False | _upsample_nearest_exact2d;_upsample_nearest_exact2d_out | shared_by_2_ops;src_only |
| aclnnUpsampleNearestExact2dBackward | 已接入 | src_scan | _upsample_nearest_exact2d_backward;_upsample_nearest_exact2d_backward.grad_input | _upsample_nearest_exact2d_backward;_upsample_nearest_exact2d_backward.grad_input | False | _upsample_nearest_exact2d_backward;_upsample_nearest_exact2d_backward_out | shared_by_2_ops;src_only |
| aclnnUpsampleNearestExact3d | 已接入 | src_scan | _upsample_nearest_exact3d;_upsample_nearest_exact3d.out | _upsample_nearest_exact3d;_upsample_nearest_exact3d.out | False | _upsample_nearest_exact3d;_upsample_nearest_exact3d_out | shared_by_2_ops;src_only |
| aclnnUpsampleNearestExact3dBackward | 已接入 | src_scan | _upsample_nearest_exact3d_backward;_upsample_nearest_exact3d_backward.grad_input | _upsample_nearest_exact3d_backward;_upsample_nearest_exact3d_backward.grad_input | False | _upsample_nearest_exact3d_backward;_upsample_nearest_exact3d_backward_out | shared_by_2_ops;src_only |
| aclnnUpsampleTrilinear3d | 已接入 | src_scan | upsample_trilinear3d;upsample_trilinear3d.out | upsample_trilinear3d;upsample_trilinear3d.out | False | upsample_trilinear3d;upsample_trilinear3d_opapi;upsample_trilinear3d_out | shared_by_2_ops;src_only |
| aclnnUpsampleTrilinear3dBackward | 已接入 | src_scan | upsample_trilinear3d_backward;upsample_trilinear3d_backward.grad_input | upsample_trilinear3d_backward;upsample_trilinear3d_backward.grad_input | False | upsample_trilinear3d_backward;upsample_trilinear3d_backward_opapi;upsample_trilinear3d_backward_out | shared_by_2_ops;src_only |
| aclnnVar | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnVarCorrection | 已接入 | src_scan | var.correction;var.correction_out |  | True | var;var_out | shared_by_2_ops;src_only |
| aclnnVarMean | 已接入 | src_scan | var_mean.correction | var_mean.correction | False | var_mean | src_only |
| aclnnWeightQuantBatchMatmul | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnWeightQuantBatchMatmulV2 | 已接入 | src_scan | npu_weight_quant_batchmatmul |  | True | npu_weight_quant_batchmatmul | src_only |
| aclnnWeightQuantBatchMatmulV3 | 已接入 | src_scan | npu_weight_quant_batchmatmul |  | True | npu_weight_quant_batchmatmul | src_only |
| aclnnWeightQuantMatmulAllReduce | 已接入 | src_scan | npu_mm_all_reduce_base |  | True | npu_mm_all_reduce_base | src_only |
| aclnnWeightQuantMatmulAllReduceAddRmsNorm | 未接入 |  |  |  | False |  | no_yaml_exec_and_no_src_scan_hit |
| aclnnXLogYScalarOther | 已接入 | yaml_exec | xlogy.OutScalar_Other;xlogy.Scalar_Other |  | True |  | shared_by_2_ops;yaml_only |
| aclnnXLogYScalarSelf | 已接入 | yaml_exec | xlogy.OutScalar_Self;xlogy.Scalar_Self |  | True |  | shared_by_2_ops;yaml_only |
| aclnnXLogYTensor | 已接入 | yaml_exec | xlogy.OutTensor;xlogy.Tensor;xlogy_.Tensor |  | True |  | shared_by_3_ops;yaml_only |
