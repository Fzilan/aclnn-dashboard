# ACLNN coverage comparison (torch-npu vs MindSpore)

**统计（基于 913 个 ACLNN API）**
- torch-npu 已接入：711 / 913（77.9%）
- mindspore 已接入：431 / 913（47.2%）
- 两者都接入：391 / 913（42.8%）
- 仅 torch-npu：320 / 913（35.0%）
- 仅 mindspore：40 / 913（4.4%）

计算公式：`占比 = 对应数量 / ACLNN 总数`
复算命令：`python3 scripts/scan/aclnn_merge_report.py --torch-npu-csv data/reports/aclnn_to_torch_npu.csv --mindspore-csv data/reports/aclnn_to_mindspore.csv --out-md data/reports/aclnn_to_all.md --out-csv data/reports/aclnn_to_all.csv`

| ACLNN API | torch-npu | via (ATen ops) | mindspore | pyboost | kbk | via (MS ops) |
|---|---|---|---|---|---|---|
| aclnnAbs | ✅ | abs;abs.out;abs_ | ✅ | ✅ | ✅ | Abs;IsInf |
| aclnnAcos | ✅ | acos;acos.out | ✅ | ✅ | ✅ | AcosExt |
| aclnnAcosh | ✅ | acosh;acosh.out | ✅ | ✅ | ✅ | AcoshExt |
| aclnnAdaLayerNorm | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnAdaLayerNormQuant | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnAdaLayerNormV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnAdaptiveAvgPool2d | ✅ | _adaptive_avg_pool2d;adaptive_avg_pool2d;adaptive_avg_pool2d.out | ✅ | ✅ | ✅ | AdaptiveAvgPool2DExt |
| aclnnAdaptiveAvgPool2dBackward | ✅ | _adaptive_avg_pool2d_backward | ✅ | ✅ | ✅ | AdaptiveAvgPool2DGradExt |
| aclnnAdaptiveAvgPool3d | ✅ | _adaptive_avg_pool3d;adaptive_avg_pool3d.out | ✅ | ✅ | ✅ | AdaptiveAvgPool3DExt;AdaptiveAvgPool3dExt |
| aclnnAdaptiveAvgPool3dBackward | ✅ | _adaptive_avg_pool3d_backward;adaptive_avg_pool3d_backward.grad_input | ✅ | ✅ | ✅ | AdaptiveAvgPool3DGradExt |
| aclnnAdaptiveMaxPool2d | ✅ | adaptive_max_pool2d;adaptive_max_pool2d.out | ✅ | ✅ | ✅ | AdaptiveMaxPool2D;AdaptiveMaxPool2d |
| aclnnAdaptiveMaxPool2dBackward | ✅ | adaptive_max_pool2d_backward;adaptive_max_pool2d_backward.grad_input | ✅ | ✅ | ✅ | AdaptiveMaxPool2DGrad;AdaptiveMaxPool2dGrad |
| aclnnAdaptiveMaxPool3d | ✅ | adaptive_max_pool3d;adaptive_max_pool3d.out | ✖️ | ✖️ | ✖️ |  |
| aclnnAdaptiveMaxPool3dBackward | ✅ | adaptive_max_pool3d_backward;adaptive_max_pool3d_backward.grad_input | ✖️ | ✖️ | ✖️ |  |
| aclnnAdd | ✅ | add.Scalar;add.Tensor;add.out | ✅ | ✅ | ✅ | Add;AddExt;Dense |
| aclnnAddLayerNorm | ✅ | npu_add_layer_norm | ✅ | ✅ | ✅ | AddLayerNormV2;AddLayernormV2 |
| aclnnAddLayerNormGrad | ✅ | npu_add_layer_norm_backward | ✅ | ✅ | ✅ | AddLayerNormGrad |
| aclnnAddLora | ✅ | npu_batch_gather_matmul;npu_batch_gather_matmul_ | ✖️ | ✖️ | ✖️ |  |
| aclnnAddRelu | ✅ | _add_relu.Tensor;_add_relu.out | ✖️ | ✖️ | ✖️ |  |
| aclnnAddRmsNorm | ✅ | npu_add_rms_norm | ✅ | ✅ | ✅ | AddRmsNorm |
| aclnnAddRmsNormCast | ✅ | npu_add_rms_norm_cast | ✖️ | ✖️ | ✖️ |  |
| aclnnAddRmsNormDynamicQuant | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnAddRmsNormDynamicQuantV2 | ✅ | npu_add_rms_norm_dynamic_quant | ✖️ | ✖️ | ✖️ |  |
| aclnnAddRmsNormQuant | ✅ | npu_add_rms_norm_quant | ✅ | ✅ | ✅ | AddRmsNormQuantV2;AddRmsnormQuantV2 |
| aclnnAddRmsNormQuantV2 | ✅ | npu_add_rms_norm_quant | ✖️ | ✖️ | ✖️ |  |
| aclnnAddbmm | ✅ | addbmm;addbmm.out;addbmm_ | ✅ | ✅ | ✅ | Addbmm |
| aclnnAddcdiv | ✅ | addcdiv;addcdiv.out | ✅ | ✅ | ✅ | AddcdivExt |
| aclnnAddcmul | ✅ | addcmul;addcmul.out | ✅ | ✅ | ✅ | AddcmulExt |
| aclnnAddmm | ✅ | addmm;addmm.out;npu_linear | ✅ | ✅ | ✅ | Addmm;Dense |
| aclnnAddmmWeightNz | ✅ | addmm;addmm.out | ✖️ | ✖️ | ✖️ |  |
| aclnnAddmv | ✅ | addmv;addmv.out;addmv_ | ✅ | ✅ | ✅ | Addmv |
| aclnnAddr | ✅ | addr;addr.out | ✖️ | ✖️ | ✖️ |  |
| aclnnAdds | ✅ | add.Scalar;add.Tensor;add.out | ✅ | ✅ | ✅ | AddScalar |
| aclnnAdvanceStep | ✅ | npu_advance_step_flashattn | ✖️ | ✖️ | ✖️ |  |
| aclnnAdvanceStepV2 | ✅ | npu_advance_step_flashattn | ✖️ | ✖️ | ✖️ |  |
| aclnnAffineGrid | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnAll | ✅ | all;all.all_out;all.dim;all.out | ✅ | ✅ | ✅ | ReduceAll |
| aclnnAllGatherMatmul | ✅ | npu_all_gather_base_mm | ✅ | ✅ | ✅ | AllGatherMatmul |
| aclnnAllGatherMatmulV2 | ✅ | npu_all_gather_base_mm;npu_all_gather_quant_mm | ✖️ | ✖️ | ✖️ |  |
| aclnnAlltoAllAllGatherBatchMatMul | ✖️ |  | ✅ | ✖️ | ✅ | AlltoAllAllGatherBatchMatMul |
| aclnnAlltoAllvGroupedMatMul | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnAmax | ✅ | amax;amax.out | ✅ | ✅ | ✅ | ReduceMax |
| aclnnAmin | ✅ | amin;amin.out | ✅ | ✅ | ✅ | ReduceMin |
| aclnnAminmax | ✅ | aminmax.out | ✖️ | ✖️ | ✖️ |  |
| aclnnAminmaxAll | ✅ | _aminmax | ✖️ | ✖️ | ✖️ |  |
| aclnnAminmaxDim | ✅ | _aminmax.dim | ✖️ | ✖️ | ✖️ |  |
| aclnnAny | ✅ | any;any.all_out;any.dim;any.out | ✅ | ✅ | ✅ | ReduceAny |
| aclnnApplyAdamW | ✅ | npu_apply_adam_w.out | ✖️ | ✖️ | ✖️ |  |
| aclnnApplyAdamWQuant | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnApplyAdamWV2 | ✅ |  | ✅ | ✅ | ✅ | AdamW;Adamw |
| aclnnApplyFusedEmaAdam | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnApplyRotaryPosEmb | ✖️ |  | ✅ | ✅ | ✅ | ApplyRotaryPosEmb |
| aclnnApplyRotaryPosEmbV2 | ✅ | npu_apply_rotary_pos_emb | ✖️ | ✖️ | ✖️ |  |
| aclnnApplyTopKTopP | ✅ | npu_top_k_top_p | ✖️ | ✖️ | ✖️ |  |
| aclnnArange | ✅ | arange;arange.out;arange.start;arange.start_out;arange.start_step | ✅ | ✅ | ✅ | Arange |
| aclnnArgMax | ✅ | argmax.out | ✅ | ✅ | ✅ | ArgMaxExt;ArgmaxExt |
| aclnnArgMin | ✅ | argmin;argmin.out | ✅ | ✅ | ✅ | ArgMinExt;ArgminExt |
| aclnnArgsort | ✖️ |  | ✅ | ✅ | ✅ |  |
| aclnnAscendAntiQuant | ✅ | npu_anti_quant | ✖️ | ✖️ | ✖️ |  |
| aclnnAscendQuant | ✅ |  | ✅ | ✅ | ✅ | QuantV2 |
| aclnnAscendQuantV3 | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnAsin | ✅ | asin;asin.out | ✅ | ✅ | ✅ | AsinExt |
| aclnnAsinh | ✅ | asinh;asinh.out | ✅ | ✅ | ✅ | AsinhExt |
| aclnnAtan | ✅ | atan;atan.out | ✅ | ✅ | ✅ | AtanExt |
| aclnnAtan2 | ✅ | atan2;atan2.out | ✅ | ✅ | ✅ | Atan2Ext |
| aclnnAtanh | ✅ | atanh;atanh.out | ✅ | ✅ | ✅ | Atanh |
| aclnnAttentionUpdate | ✅ | npu_attention_update | ✖️ | ✖️ | ✖️ |  |
| aclnnAvgPool2d | ✅ | avg_pool2d;avg_pool2d.out | ✅ | ✅ | ✅ | AvgPool1D;AvgPool2D;AvgPool2d |
| aclnnAvgPool2dBackward | ✅ | avg_pool2d_backward;avg_pool2d_backward.grad_input | ✅ | ✅ | ✅ | AvgPool2DGrad;AvgPool2dGrad |
| aclnnAvgPool3d | ✅ | avg_pool3d;avg_pool3d.out | ✅ | ✅ | ✅ | AvgPool3DExt;AvgPool3dExt |
| aclnnAvgPool3dBackward | ✅ | avg_pool3d_backward;avg_pool3d_backward.grad_input | ✅ | ✅ | ✅ | AvgPool3DGradExt;AvgPool3dGradExt |
| aclnnBackgroundReplace | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnBaddbmm | ✅ | baddbmm;baddbmm.out;baddbmm_ | ✅ | ✅ | ✅ | Baddbmm |
| aclnnBatchMatMul | ✅ | affine_grid_generator_backward;bmm;bmm.out | ✅ | ✅ | ✅ | BatchMatMulExt;BmmExt |
| aclnnBatchMatMulReduceScatterAlltoAll | ✖️ |  | ✅ | ✖️ | ✅ | BatchMatMulReduceScatterAlltoAll |
| aclnnBatchMatMulWeightNz | ✅ | bmm;bmm.out | ✖️ | ✖️ | ✖️ |  |
| aclnnBatchMatmulQuant | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnBatchNorm | ✅ | native_batch_norm;native_batch_norm.out | ✅ | ✅ | ✅ | BatchNormExt |
| aclnnBatchNormBackward | ✅ | native_batch_norm_backward | ✅ | ✅ | ✅ | BatchNormGradExt |
| aclnnBatchNormElemt | ✅ | batch_norm_elemt;batch_norm_elemt.out | ✅ | ✅ | ✅ | BatchNormElemt |
| aclnnBatchNormElemtBackward | ✅ | batch_norm_backward_elemt | ✅ | ✅ | ✅ | BatchNormElemtGrad |
| aclnnBatchNormGatherStatsWithCounts | ✅ | batch_norm_gather_stats_with_counts | ✅ | ✅ | ✅ | BatchNormGatherStatsWithCounts |
| aclnnBatchNormReduce | ✅ | batch_norm_reduce | ✖️ | ✖️ | ✖️ |  |
| aclnnBatchNormReduceBackward | ✅ | batch_norm_backward_reduce | ✅ | ✅ | ✅ | BatchNormReduceGrad |
| aclnnBatchNormStats | ✅ | batch_norm_stats | ✅ | ✅ | ✅ | BatchNormStats |
| aclnnBernoulli | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnBernoulliTensor | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnBidirectionLSTM | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnBidirectionLSTMV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnBinaryCrossEntropy | ✅ | binary_cross_entropy;binary_cross_entropy.out | ✅ | ✅ | ✅ | BinaryCrossEntropy |
| aclnnBinaryCrossEntropyBackward | ✅ | binary_cross_entropy_backward;binary_cross_entropy_backward.grad_input | ✅ | ✅ | ✅ | BinaryCrossEntropyGrad |
| aclnnBinaryCrossEntropyWithLogits | ✅ | binary_cross_entropy_with_logits | ✅ | ✅ | ✅ | BCEWithLogitsLoss;BinaryCrossEntropyWithLogits |
| aclnnBinaryCrossEntropyWithLogitsBackward | ✅ | npu_binary_cross_entropy_with_logits_backward | ✅ | ✅ | ✅ | BinaryCrossEntropyWithLogitsBackward |
| aclnnBinaryCrossEntropyWithLogitsTargetBackward | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnBincount | ✅ | bincount | ✅ | ✅ | ✅ | BincountExt |
| aclnnBitwiseAndScalar | ✅ | bitwise_and.Scalar;bitwise_and.Scalar_out;bitwise_and.Tensor;bitwise_and.Tensor_out | ✅ | ✅ | ✅ | BitwiseAndScalar |
| aclnnBitwiseAndTensor | ✅ | bitwise_and.Scalar;bitwise_and.Scalar_out;bitwise_and.Tensor;bitwise_and.Tensor_out | ✅ | ✅ | ✅ | BitwiseAndTensor |
| aclnnBitwiseNot | ✅ | bitwise_not;bitwise_not.out;bitwise_not_ | ✅ | ✅ | ✅ | BitwiseNot |
| aclnnBitwiseOrScalar | ✅ | bitwise_or.Scalar;bitwise_or.Scalar_out;bitwise_or.Tensor;bitwise_or.Tensor_out | ✅ | ✅ | ✅ | BitwiseOrScalar |
| aclnnBitwiseOrTensor | ✅ | bitwise_or.Scalar;bitwise_or.Scalar_out;bitwise_or.Tensor;bitwise_or.Tensor_out | ✅ | ✅ | ✅ | BitwiseOrTensor |
| aclnnBitwiseXorScalar | ✅ | bitwise_xor.Scalar;bitwise_xor.Scalar_out;bitwise_xor.Tensor;bitwise_xor.Tensor_out | ✅ | ✅ | ✅ | BitwiseXorScalar |
| aclnnBitwiseXorTensor | ✅ | bitwise_xor.Scalar;bitwise_xor.Scalar_out;bitwise_xor.Tensor;bitwise_xor.Tensor_out | ✅ | ✅ | ✅ | BitwiseXorTensor |
| aclnnBlendImagesCustom | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnCalculateConvolutionWeightSize | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnCalculateMatmulWeightSize | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnCalculateMatmulWeightSizeV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnCast | ✅ | _npu_dtype_cast;npu_dtype_cast | ✅ | ✅ | ✅ | Cast;MultiScaleDeformableAttn;MultiScaleDeformableAttnGrad |
| aclnnCat | ✅ | cat;cat.names;cat.names_out;cat.out | ✅ | ✅ | ✅ | Concat |
| aclnnCeil | ✅ | ceil;ceil.out | ✅ | ✅ | ✅ | Ceil |
| aclnnCelu | ✅ | celu | ✖️ | ✖️ | ✖️ |  |
| aclnnChamferDistanceBackward | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnChannelShuffle | ✅ | channel_shuffle | ✖️ | ✖️ | ✖️ |  |
| aclnnCircularPad2d | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnCircularPad2dBackward | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnCircularPad3d | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnCircularPad3dBackward | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnClamp | ✅ | clamp;clamp.out;clamp_ | ✅ | ✅ | ✅ | ClampScalar;InplaceClampScalar |
| aclnnClampMax | ✅ | clamp_max;clamp_max.out | ✖️ | ✖️ | ✖️ |  |
| aclnnClampMaxTensor | ✅ | clamp_max.Tensor;clamp_max.Tensor_out | ✖️ | ✖️ | ✖️ |  |
| aclnnClampMin | ✅ | clamp_min;clamp_min.out;clamp_min_ | ✅ | ✅ | ✅ | ClampMin |
| aclnnClampMinTensor | ✅ | clamp_min.Tensor;clamp_min.Tensor_out | ✖️ | ✖️ | ✖️ |  |
| aclnnClampTensor | ✅ | clamp.Tensor;clamp.Tensor_out;clamp_.Tensor | ✅ | ✅ | ✅ | ClampTensor;InplaceClampTensor |
| aclnnClippedSwiglu | ✅ | npu_clipped_swiglu | ✖️ | ✖️ | ✖️ |  |
| aclnnComplex | ✅ | complex;complex.out | ✖️ | ✖️ | ✖️ |  |
| aclnnConstantPadNd | ✅ | constant_pad_nd | ✅ | ✅ | ✅ | ConstantPadND;ConvolutionStr |
| aclnnConvDepthwise2d | ✅ | _conv_depthwise2d;_conv_depthwise2d.out | ✖️ | ✖️ | ✖️ |  |
| aclnnConvTbc | ✅ | conv_tbc | ✖️ | ✖️ | ✖️ |  |
| aclnnConvTbcBackward | ✅ | conv_tbc_backward | ✖️ | ✖️ | ✖️ |  |
| aclnnConvertWeightToINT4Pack | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnConvolution | ✅ | _convolution;_nnpack_spatial_convolution;_slow_conv2d_forward;_slow_conv2d_forward.output;convolution_overrideable;slow_conv3d_forward;slow_conv3d_forward.output;slow_conv_dilated2d;slow_conv_transpose2d;slow_conv_transpose2d.out | ✅ | ✅ | ✅ | Conv1DExt;Conv2DExt;Conv3DExt;ConvTranspose2D;Convolution;ConvolutionStr |
| aclnnConvolutionBackward | ✅ | _slow_conv2d_backward.output_mask;convolution_backward;convolution_backward_overrideable;slow_conv_dilated2d_backward;slow_conv_transpose2d_backward | ✅ | ✅ | ✅ | ConvolutionGrad;ConvolutionStrGrad |
| aclnnCos | ✅ | cos;cos.out | ✅ | ✅ | ✅ | Cos |
| aclnnCosh | ✅ | cosh;cosh.out | ✅ | ✅ | ✅ | Cosh |
| aclnnCrossEntropyLoss | ✅ | npu_cross_entropy_loss | ✅ | ✅ | ✅ | CrossEntropyLoss |
| aclnnCrossEntropyLossGrad | ✅ | npu_cross_entropy_loss_backward | ✅ | ✅ | ✅ | CrossEntropyLossGrad |
| aclnnCtcLoss | ✅ | _ctc_loss | ✖️ | ✖️ | ✖️ |  |
| aclnnCtcLossBackward | ✅ | _ctc_loss_backward | ✖️ | ✖️ | ✖️ |  |
| aclnnCummax | ✅ | _cummax_helper | ✅ | ✅ | ✅ | Cummax |
| aclnnCummin | ✅ | _cummin_helper | ✅ | ✅ | ✅ | CumminExt |
| aclnnCumprod | ✅ | cumprod.out | ✖️ | ✖️ | ✖️ |  |
| aclnnCumsum | ✅ | cumsum;cumsum.dimname_out;cumsum.out | ✅ | ✅ | ✅ | CumsumExt |
| aclnnCumsumV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnDeepNorm | ✅ | npu_deep_norm | ✖️ | ✖️ | ✖️ |  |
| aclnnDeepNormGrad | ✅ | npu_deep_norm_backward | ✖️ | ✖️ | ✖️ |  |
| aclnnDeformableConv2d | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnDequantBias | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnDequantRopeQuantKvcache | ✅ | npu_dequant_rope_quant_kvcache;npu_rope_quant_kvcache | ✖️ | ✖️ | ✖️ |  |
| aclnnDequantSwigluQuant | ✅ | npu_dequant_swiglu_quant | ✅ | ✅ | ✅ | DequantSwigluQuant |
| aclnnDiag | ✖️ |  | ✅ | ✅ | ✅ | DiagExt |
| aclnnDiagFlat | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnDigamma | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnDistributeBarrier | ✅ | _npu_distribute_barrier | ✖️ | ✖️ | ✖️ |  |
| aclnnDistributeBarrierV2 | ✅ | _npu_distribute_barrier | ✖️ | ✖️ | ✖️ |  |
| aclnnDiv | ✅ | div.Scalar;div.Scalar_mode;div.Tensor;div.Tensor_mode;div.out;div.out_mode | ✅ | ✅ | ✅ | Div;RealDiv |
| aclnnDivMod | ✅ | div.Scalar;div.Scalar_mode;div.Tensor;div.Tensor_mode;div.out;div.out_mode | ✅ | ✅ | ✅ | DivMod;Divmod |
| aclnnDivMods | ✅ | div.Scalar;div.Scalar_mode;div.Tensor;div.Tensor_mode;div.out;div.out_mode | ✅ | ✅ | ✅ | DivMods;Divmods |
| aclnnDivs | ✅ | div.Scalar;div.Scalar_mode;div.Tensor;div.Tensor_mode;div.out;div.out_mode | ✅ | ✅ | ✅ | Divs |
| aclnnDot | ✅ | dot;dot.out;vdot;vdot.out | ✅ | ✅ | ✅ | Dot |
| aclnnDropout | ✖️ |  | ✅ | ✖️ | ✅ | DropoutExt |
| aclnnDropoutBackward | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnDropoutDoMask | ✅ | _npu_dropout;native_dropout;native_dropout_backward;npu_dropout_backward | ✅ | ✅ | ✅ | DropoutDoMaskExt;DropoutExt;DropoutGradExt |
| aclnnDropoutGenMask | ✅ | npu_dropout_gen_mask | ✖️ | ✖️ | ✖️ |  |
| aclnnDropoutGenMaskV2 | ✅ | _npu_dropout;_npu_dropout_gen_mask.Tensor;native_dropout | ✅ | ✅ | ✅ | DropoutExt;DropoutGenMaskExt |
| aclnnDropoutGenMaskV2Tensor | ✅ | _npu_dropout | ✖️ | ✖️ | ✖️ |  |
| aclnnDynamicBlockQuant | ✅ | npu_dynamic_block_quant | ✖️ | ✖️ | ✖️ |  |
| aclnnDynamicQuant | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnDynamicQuantV2 | ✅ | npu_dynamic_quant;npu_dynamic_quant_asymmetric | ✖️ | ✖️ | ✖️ |  |
| aclnnEinsum | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnElu | ✅ | elu;elu.out | ✅ | ✅ | ✅ | EluExt |
| aclnnEluBackward | ✅ | elu_backward;elu_backward.grad_input | ✅ | ✅ | ✅ | EluGradExt |
| aclnnEmbedding | ✅ |  | ✅ | ✅ | ✅ | Embedding |
| aclnnEmbeddingBag | ✅ | _embedding_bag;_embedding_bag_forward_only | ✖️ | ✖️ | ✖️ |  |
| aclnnEmbeddingDenseBackward | ✅ | embedding_dense_backward | ✅ | ✅ | ✅ | EmbeddingDenseBackward |
| aclnnEmbeddingRenorm | ✅ | embedding_renorm_ | ✅ | ✅ | ✅ | Embedding |
| aclnnEqScalar | ✅ | eq.Scalar;eq.Scalar_out;eq.Tensor;eq.Tensor_out | ✅ | ✅ | ✅ | EqScalar;IsInf;Isinf |
| aclnnEqTensor | ✅ | eq.Scalar;eq.Scalar_out;eq.Tensor;eq.Tensor_out | ✅ | ✅ | ✅ | Equal |
| aclnnEqual | ✅ | equal | ✅ | ✅ | ✅ | EqualExt |
| aclnnErf | ✅ | erf;erf.out | ✅ | ✅ | ✅ | Erf |
| aclnnErfc | ✅ | erfc;erfc.out | ✅ | ✅ | ✅ | Erfc |
| aclnnErfinv | ✅ | erfinv;erfinv.out | ✅ | ✅ | ✅ | Erfinv |
| aclnnExp | ✅ | exp;exp.out | ✅ | ✅ | ✅ | Exp |
| aclnnExp2 | ✅ | exp2;exp2.out | ✅ | ✅ | ✅ | Exp2 |
| aclnnExpSegsum | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnExpSegsumBackward | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnExpand | ✖️ |  | ✅ | ✖️ | ✅ | BroadcastTo;ExpandAs;L1LossBackwardExt;L1LossExt;MSELossExt;MSELossGradExt |
| aclnnExpandIntoJaggedPermute | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnExpm1 | ✅ | expm1;expm1.out | ✅ | ✅ | ✅ | Expm1 |
| aclnnEye | ✅ | eye;eye.m;eye.m_out;eye.out | ✅ | ✅ | ✅ | Eye |
| aclnnFFN | ✖️ |  | ✅ | ✅ | ✅ | FFNExt;FfnExt |
| aclnnFFNV2 | ✅ | npu_ffn | ✖️ | ✖️ | ✖️ |  |
| aclnnFFNV3 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnFakeQuantPerChannelAffineCachemask | ✅ | fake_quantize_per_channel_affine_cachemask | ✖️ | ✖️ | ✖️ |  |
| aclnnFakeQuantPerTensorAffineCachemask | ✅ | _fake_quantize_per_tensor_affine_cachemask_tensor_qparams | ✖️ | ✖️ | ✖️ |  |
| aclnnFastBatchNormBackward | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnFastGelu | ✅ | npu_fast_gelu | ✖️ | ✖️ | ✖️ |  |
| aclnnFastGeluBackward | ✅ | npu_fast_gelu_backward | ✖️ | ✖️ | ✖️ |  |
| aclnnFatreluMul | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnFinalize | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnFlashAttentionScore | ✖️ |  | ✅ | ✅ | ✅ | FlashAttentionScore |
| aclnnFlashAttentionScoreGrad | ✖️ |  | ✅ | ✅ | ✅ | FlashAttentionScoreGrad |
| aclnnFlashAttentionScoreGradV2 | ✖️ |  | ✅ | ✅ | ✖️ | SpeedFusionAttentionGrad |
| aclnnFlashAttentionScoreGradV3 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnFlashAttentionScoreV2 | ✖️ |  | ✅ | ✅ | ✖️ | SpeedFusionAttention |
| aclnnFlashAttentionScoreV3 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnFlashAttentionUnpaddingScoreGrad | ✖️ |  | ✅ | ✅ | ✖️ | FlashAttentionScoreGrad |
| aclnnFlashAttentionUnpaddingScoreGradV2 | ✖️ |  | ✅ | ✅ | ✖️ | SpeedFusionAttentionGrad |
| aclnnFlashAttentionUnpaddingScoreGradV3 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnFlashAttentionUnpaddingScoreGradV4 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnFlashAttentionUnpaddingScoreGradV5 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnFlashAttentionVarLenScore | ✖️ |  | ✅ | ✅ | ✖️ | FlashAttentionScore |
| aclnnFlashAttentionVarLenScoreV2 | ✖️ |  | ✅ | ✅ | ✖️ | SpeedFusionAttention |
| aclnnFlashAttentionVarLenScoreV3 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnFlashAttentionVarLenScoreV4 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnFlashAttentionVarLenScoreV5 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnFlatQuant | ✅ | npu_kronecker_quant | ✖️ | ✖️ | ✖️ |  |
| aclnnFlatten | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnFlip | ✅ | flip | ✅ | ✅ | ✅ | ReverseV2 |
| aclnnFloor | ✅ | floor;floor.out | ✅ | ✅ | ✅ | Floor |
| aclnnFloorDivide | ✅ | floor_divide;floor_divide.Scalar;floor_divide.out | ✅ | ✅ | ✅ | FloorDiv |
| aclnnFloorDivides | ✅ | floor_divide;floor_divide.Scalar;floor_divide.out | ✅ | ✅ | ✅ | FloorDivScalar |
| aclnnFmodScalar | ✅ | fmod.Scalar;fmod.Scalar_out | ✅ | ✅ | ✅ | FmodScalar |
| aclnnFmodTensor | ✅ | fmod.Tensor;fmod.Tensor_out | ✅ | ✅ | ✅ | FmodTensor |
| aclnnForeachAbs | ✅ | _foreach_abs;_foreach_abs_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachAcos | ✅ | _foreach_acos;_foreach_acos_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachAddList | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachAddListV2 | ✅ | _foreach_add.List;_foreach_add.Scalar;_foreach_add.ScalarList;_foreach_add_.List;_foreach_add_.Scalar;_foreach_add_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachAddScalar | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachAddScalarList | ✅ | _foreach_add.List;_foreach_add.Scalar;_foreach_add.ScalarList;_foreach_add_.List;_foreach_add_.Scalar;_foreach_add_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachAddScalarV2 | ✅ | _foreach_add.List;_foreach_add.Scalar;_foreach_add.ScalarList;_foreach_add_.List;_foreach_add_.Scalar;_foreach_add_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachAddcdivList | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachAddcdivScalar | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachAddcdivScalarList | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachAddcdivScalarV2 | ✅ | _foreach_addcdiv.Scalar;_foreach_addcdiv.ScalarList;_foreach_addcdiv.Tensor;_foreach_addcdiv_.Scalar;_foreach_addcdiv_.ScalarList;_foreach_addcdiv_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachAddcmulList | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachAddcmulScalar | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachAddcmulScalarList | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachAddcmulScalarV2 | ✅ | _foreach_addcmul.Scalar;_foreach_addcmul.ScalarList;_foreach_addcmul.Tensor;_foreach_addcmul_.Scalar;_foreach_addcmul_.ScalarList;_foreach_addcmul_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachAsin | ✅ | _foreach_asin;_foreach_asin_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachAtan | ✅ | _foreach_atan;_foreach_atan_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachCopy | ✅ | _foreach_copy_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachCos | ✅ | _foreach_cos;_foreach_cos_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachCosh | ✅ | _foreach_cosh;_foreach_cosh_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachDivList | ✅ | _foreach_div.List;_foreach_div.Scalar;_foreach_div.ScalarList;_foreach_div_.List;_foreach_div_.Scalar;_foreach_div_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachDivScalar | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachDivScalarList | ✅ | _foreach_div.List;_foreach_div.Scalar;_foreach_div.ScalarList;_foreach_div_.List;_foreach_div_.Scalar;_foreach_div_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachDivScalarV2 | ✅ | _foreach_div.List;_foreach_div.Scalar;_foreach_div.ScalarList;_foreach_div_.List;_foreach_div_.Scalar;_foreach_div_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachErf | ✅ | _foreach_erf;_foreach_erf_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachErfc | ✅ | _foreach_erfc;_foreach_erfc_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachExp | ✅ | _foreach_exp;_foreach_exp_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachExpm1 | ✅ | _foreach_expm1;_foreach_expm1_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachLerpList | ✅ | _foreach_lerp.List;_foreach_lerp.Scalar;_foreach_lerp_.List;_foreach_lerp_.Scalar | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachLerpScalar | ✅ | _foreach_lerp.List;_foreach_lerp.Scalar;_foreach_lerp_.List;_foreach_lerp_.Scalar | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachLog | ✅ | _foreach_log;_foreach_log_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachLog10 | ✅ | _foreach_log10;_foreach_log10_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachLog1p | ✅ | _foreach_log1p;_foreach_log1p_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachLog2 | ✅ | _foreach_log2;_foreach_log2_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachMaximumList | ✅ | _foreach_maximum.List;_foreach_maximum.Scalar;_foreach_maximum.ScalarList;_foreach_maximum_.List;_foreach_maximum_.Scalar;_foreach_maximum_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachMaximumScalar | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachMaximumScalarList | ✅ | _foreach_maximum.List;_foreach_maximum.Scalar;_foreach_maximum.ScalarList;_foreach_maximum_.List;_foreach_maximum_.Scalar;_foreach_maximum_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachMaximumScalarV2 | ✅ | _foreach_maximum.List;_foreach_maximum.Scalar;_foreach_maximum.ScalarList;_foreach_maximum_.List;_foreach_maximum_.Scalar;_foreach_maximum_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachMinimumList | ✅ | _foreach_minimum.List;_foreach_minimum.Scalar;_foreach_minimum.ScalarList;_foreach_minimum_.List;_foreach_minimum_.Scalar;_foreach_minimum_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachMinimumScalar | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachMinimumScalarList | ✅ | _foreach_minimum.List;_foreach_minimum.Scalar;_foreach_minimum.ScalarList;_foreach_minimum_.List;_foreach_minimum_.Scalar;_foreach_minimum_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachMinimumScalarV2 | ✅ | _foreach_minimum.List;_foreach_minimum.Scalar;_foreach_minimum.ScalarList;_foreach_minimum_.List;_foreach_minimum_.Scalar;_foreach_minimum_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachMulList | ✅ | _foreach_mul.List;_foreach_mul.Scalar;_foreach_mul.ScalarList;_foreach_mul_.List;_foreach_mul_.Scalar;_foreach_mul_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachMulScalar | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachMulScalarList | ✅ | _foreach_mul.List;_foreach_mul.Scalar;_foreach_mul.ScalarList;_foreach_mul_.List;_foreach_mul_.Scalar;_foreach_mul_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachMulScalarV2 | ✅ | _foreach_mul.List;_foreach_mul.Scalar;_foreach_mul.ScalarList;_foreach_mul_.List;_foreach_mul_.Scalar;_foreach_mul_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachNeg | ✅ | _foreach_neg;_foreach_neg_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachNonFiniteCheckAndUnscale | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachNorm | ✅ | _foreach_norm.Scalar | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachPowList | ✅ | _foreach_pow.List;_foreach_pow.Scalar;_foreach_pow.ScalarAndTensor;_foreach_pow.ScalarList;_foreach_pow_.List;_foreach_pow_.Scalar;_foreach_pow_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachPowScalar | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachPowScalarAndTensor | ✅ | _foreach_pow.List;_foreach_pow.Scalar;_foreach_pow.ScalarAndTensor;_foreach_pow.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachPowScalarList | ✅ | _foreach_pow.List;_foreach_pow.Scalar;_foreach_pow.ScalarAndTensor;_foreach_pow.ScalarList;_foreach_pow_.List;_foreach_pow_.Scalar;_foreach_pow_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachPowScalarV2 | ✅ | _foreach_pow.List;_foreach_pow.Scalar;_foreach_pow.ScalarAndTensor;_foreach_pow.ScalarList;_foreach_pow_.List;_foreach_pow_.Scalar;_foreach_pow_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachReciprocal | ✅ | _foreach_reciprocal;_foreach_reciprocal_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachRoundOffNumber | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachRoundOffNumberV2 | ✅ | _foreach_ceil;_foreach_ceil_;_foreach_floor;_foreach_floor_;_foreach_frac;_foreach_frac_;_foreach_round;_foreach_round_;_foreach_trunc;_foreach_trunc_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachSigmoid | ✅ | _foreach_sigmoid;_foreach_sigmoid_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachSign | ✅ | _foreach_sign;_foreach_sign_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachSin | ✅ | _foreach_sin;_foreach_sin_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachSinh | ✅ | _foreach_sinh;_foreach_sinh_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachSqrt | ✅ | _foreach_sqrt;_foreach_sqrt_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachSubList | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachSubListV2 | ✅ | _foreach_sub.List;_foreach_sub.Scalar;_foreach_sub.ScalarList;_foreach_sub_.List;_foreach_sub_.Scalar;_foreach_sub_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachSubScalar | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachSubScalarList | ✅ | _foreach_sub.List;_foreach_sub.Scalar;_foreach_sub.ScalarList;_foreach_sub_.List;_foreach_sub_.Scalar;_foreach_sub_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachSubScalarV2 | ✅ | _foreach_sub.List;_foreach_sub.Scalar;_foreach_sub.ScalarList;_foreach_sub_.List;_foreach_sub_.Scalar;_foreach_sub_.ScalarList | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachTan | ✅ | _foreach_tan;_foreach_tan_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachTanh | ✅ | _foreach_tanh;_foreach_tanh_ | ✖️ | ✖️ | ✖️ |  |
| aclnnForeachZeroInplace | ✅ | _foreach_zero_ | ✖️ | ✖️ | ✖️ |  |
| aclnnFrac | ✅ | frac;frac.out | ✅ | ✅ | ✅ | Frac |
| aclnnFusedCrossEntropyLossWithMaxSum | ✅ | fused_cross_entropy_loss_with_max_sum | ✖️ | ✖️ | ✖️ |  |
| aclnnFusedInferAttentionScore | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnFusedInferAttentionScoreV2 | ✖️ |  | ✅ | ✅ | ✅ | FusedInferAttentionScore |
| aclnnFusedInferAttentionScoreV3 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnFusedInferAttentionScoreV4 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnFusedLinearCrossEntropyLossGrad | ✅ | fused_linear_cross_entropy_loss_with_max_sum_grad | ✖️ | ✖️ | ✖️ |  |
| aclnnFusedLinearOnlineMaxSum | ✅ | fused_linear_online_max_sum | ✖️ | ✖️ | ✖️ |  |
| aclnnGather | ✅ | gather;gather.dimname;gather.dimname_out;gather.out | ✅ | ✅ | ✅ | GatherD |
| aclnnGatherNd | ✖️ |  | ✅ | ✅ | ✅ | GatherNdExt |
| aclnnGatherPaKvCache | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnGatherV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnGatherV3 | ✅ | npu_gather_sparse_index | ✖️ | ✖️ | ✖️ |  |
| aclnnGcd | ✅ | gcd.out | ✅ | ✅ | ✅ | Gcd |
| aclnnGeGlu | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnGeGluBackward | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnGeGluV3 | ✅ | npu_geglu | ✖️ | ✖️ | ✖️ |  |
| aclnnGeGluV3Backward | ✅ | npu_geglu_grad | ✖️ | ✖️ | ✖️ |  |
| aclnnGeScalar | ✅ | ge.Scalar;ge.Scalar_out;ge.Tensor;ge.Tensor_out | ✅ | ✅ | ✅ | GreaterEqualScalar |
| aclnnGeTensor | ✅ | ge.Scalar;ge.Scalar_out;ge.Tensor;ge.Tensor_out | ✅ | ✅ | ✅ | GreaterEqual |
| aclnnGelu | ✅ | gelu.out | ✅ | ✅ | ✅ | GeLU |
| aclnnGeluBackward | ✅ | gelu_backward | ✅ | ✅ | ✅ | GeLUGrad;GeluGrad |
| aclnnGeluBackwardV2 | ✅ | gelu_backward;npu_gelu_backward | ✅ | ✅ | ✅ | GeluGradExt |
| aclnnGeluMul | ✅ | npu_gelu_mul | ✖️ | ✖️ | ✖️ |  |
| aclnnGeluV2 | ✅ | gelu.out;npu_gelu | ✅ | ✅ | ✅ | GeluExt |
| aclnnGemm | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnGemmaRmsNorm | ✅ | npu_gemma_rms_norm | ✖️ | ✖️ | ✖️ |  |
| aclnnGer | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnGlobalAveragePool | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnGlobalMaxPool | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnGlu | ✅ | glu;glu.out | ✅ | ✅ | ✅ | GLU |
| aclnnGluBackward | ✅ | glu_backward;glu_backward.grad_input | ✅ | ✅ | ✅ | GluGrad |
| aclnnGridSampler2D | ✅ | grid_sampler_2d | ✅ | ✅ | ✅ | GridSampler2D |
| aclnnGridSampler2DBackward | ✅ | grid_sampler_2d_backward | ✅ | ✅ | ✅ | GridSampler2DGrad;GridSampler2dGrad |
| aclnnGridSampler3D | ✅ | grid_sampler_3d | ✅ | ✅ | ✅ | GridSampler3D |
| aclnnGridSampler3DBackward | ✅ | grid_sampler_3d_backward | ✅ | ✅ | ✅ | GridSampler3DGrad;GridSampler3dGrad |
| aclnnGroupNorm | ✅ | native_group_norm | ✅ | ✅ | ✅ | GroupNorm |
| aclnnGroupNormBackward | ✅ | native_group_norm_backward | ✅ | ✅ | ✅ | GroupNormGrad |
| aclnnGroupNormSilu | ✅ | npu_group_norm_silu | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupNormSiluV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupNormSwish | ✅ | npu_group_norm_swish | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupNormSwishGrad | ✅ | npu_group_norm_swish_grad | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupQuant | ✅ | npu_group_quant | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupedBiasAddGrad | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupedBiasAddGradV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupedMatMulAllReduce | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupedMatMulAlltoAllv | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupedMatmul | ✅ | npu_grouped_matmul;npu_grouped_matmul.List | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupedMatmulAdd | ✅ | npu_grouped_matmul_add | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupedMatmulFinalizeRouting | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupedMatmulFinalizeRoutingV2 | ✅ | npu_grouped_matmul_finalize_routing | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupedMatmulFinalizeRoutingV3 | ✅ | npu_grouped_matmul_finalize_routing | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupedMatmulFinalizeRoutingWeightNz | ✅ | npu_grouped_matmul_finalize_routing | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupedMatmulFinalizeRoutingWeightNzV2 | ✅ | npu_grouped_matmul_finalize_routing | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupedMatmulSwigluQuant | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupedMatmulSwigluQuantV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupedMatmulSwigluQuantWeightNZ | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupedMatmulV2 | ✖️ |  | ✅ | ✅ | ✅ | GroupedMatmulV2 |
| aclnnGroupedMatmulV3 | ✖️ |  | ✅ | ✅ | ✅ | GroupedMatmul |
| aclnnGroupedMatmulV4 | ✅ | npu_grouped_matmul;npu_grouped_matmul.List | ✅ | ✅ | ✅ | GroupedMatmulV4 |
| aclnnGroupedMatmulV5 | ✅ | npu_grouped_matmul;npu_grouped_matmul.List | ✖️ | ✖️ | ✖️ |  |
| aclnnGroupedMatmulWeightNz | ✅ | npu_grouped_matmul;npu_grouped_matmul.List | ✖️ | ✖️ | ✖️ |  |
| aclnnGtScalar | ✅ | gt.Scalar;gt.Scalar_out;gt.Tensor;gt.Tensor_out | ✖️ | ✖️ | ✖️ |  |
| aclnnGtTensor | ✅ | gt.Scalar;gt.Scalar_out;gt.Tensor;gt.Tensor_out | ✅ | ✅ | ✅ | Greater |
| aclnnHardshrink | ✅ | hardshrink;hardshrink.out | ✅ | ✅ | ✅ | HShrink;Hshrink |
| aclnnHardshrinkBackward | ✅ | hardshrink_backward;hardshrink_backward.grad_input | ✅ | ✅ | ✅ | HShrinkGrad;HshrinkGrad |
| aclnnHardsigmoid | ✅ | hardsigmoid;hardsigmoid.out | ✅ | ✅ | ✅ | HSigmoid |
| aclnnHardsigmoidBackward | ✅ | hardsigmoid_backward | ✅ | ✅ | ✅ | HSigmoidGrad |
| aclnnHardswish | ✅ | hardswish;hardswish.out | ✅ | ✅ | ✅ | HSwish |
| aclnnHardswishBackward | ✅ | hardswish_backward | ✅ | ✅ | ✅ | HSwishGrad |
| aclnnHardswishBackwardV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnHardtanh | ✅ | hardtanh;hardtanh.out | ✅ | ✅ | ✅ | Hardtanh |
| aclnnHardtanhBackward | ✅ | hardtanh_backward;hardtanh_backward.grad_input | ✅ | ✅ | ✅ | HardtanhGrad |
| aclnnHeaviside | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnHistc | ✅ | histc;histc.out | ✅ | ✅ | ✅ | HistcExt |
| aclnnIm2col | ✅ | im2col;im2col.out | ✅ | ✅ | ✅ | Col2ImGrad;Im2ColExt |
| aclnnIm2colBackward | ✅ | col2im;col2im.out | ✅ | ✅ | ✅ | Col2ImExt |
| aclnnIncreFlashAttention | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnIncreFlashAttentionV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnIncreFlashAttentionV3 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnIncreFlashAttentionV4 | ✖️ |  | ✅ | ✅ | ✅ | IncreFlashAttention |
| aclnnIndex | ✅ | index.Tensor | ✅ | ✅ | ✅ | InnerIndex |
| aclnnIndexAdd | ✅ | index_add;index_add.dimname;index_add.out | ✅ | ✅ | ✅ | IndexAddExt;InplaceIndexAdd;InplaceIndexAddExt |
| aclnnIndexCopy | ✅ | index_copy;index_copy.out | ✖️ | ✖️ | ✖️ |  |
| aclnnIndexFill | ✅ | index_fill.int_Scalar;index_fill.int_Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnIndexFillTensor | ✅ | index_fill.int_Scalar;index_fill.int_Tensor | ✅ | ✅ | ✅ | IndexFillScalar;IndexFillTensor |
| aclnnIndexPutImpl | ✅ | _index_put_impl_;index_put;index_put_ | ✅ | ✅ | ✅ | InnerInplaceIndexPut |
| aclnnIndexSelect | ✅ | index_select;index_select.dimname;index_select.dimname_out;index_select.out | ✅ | ✅ | ✅ | IndexSelect |
| aclnnInit | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceAcos | ✅ | acos_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceAcosh | ✅ | acosh_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceAdd | ✅ | add_.Scalar;add_.Tensor | ✅ | ✅ | ✅ | InplaceAddExt |
| aclnnInplaceAddRelu | ✅ | _add_relu_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceAddRmsNorm | ✅ | npu_add_rms_norm_v2;npu_add_rms_norm_v2_functional | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceAddbmm | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceAddcdiv | ✅ | addcdiv_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceAddcmul | ✅ | addcmul_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceAddmm | ✅ | addmm_ | ✅ | ✅ | ✅ | InplaceAddmm |
| aclnnInplaceAddr | ✅ | addr_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceAdds | ✅ | add_.Scalar;add_.Tensor | ✅ | ✅ | ✅ | InplaceAddsExt |
| aclnnInplaceAsin | ✅ | asin_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceAsinh | ✅ | asinh_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceAtan | ✅ | atan_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceAtan2 | ✅ | atan2_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceAtanh | ✅ | atanh_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceAttentionWorkerScheduler | ✅ | attention_worker_scheduler;attention_worker_scheduler_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceBaddbmm | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceBernoulli | ✅ | bernoulli;bernoulli.out;bernoulli.p;bernoulli_.Tensor;bernoulli_.float | ✅ | ✅ | ✅ | InplaceBernoulliScalar |
| aclnnInplaceBernoulliTensor | ✅ | bernoulli;bernoulli.out;bernoulli.p;bernoulli_.Tensor;bernoulli_.float | ✅ | ✅ | ✅ | BernoulliExt;InplaceBernoulliTensor |
| aclnnInplaceBitwiseAndScalar | ✅ | bitwise_and_.Scalar;bitwise_and_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceBitwiseAndTensor | ✅ | bitwise_and_.Scalar;bitwise_and_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceBitwiseOrScalar | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceBitwiseOrTensor | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceBitwiseXorScalar | ✅ | bitwise_xor_.Scalar;bitwise_xor_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceBitwiseXorTensor | ✅ | bitwise_xor_.Scalar;bitwise_xor_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceCeil | ✅ | ceil_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceCelu | ✅ | celu_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceClampMax | ✅ | clamp_max_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceClampMaxTensor | ✅ | clamp_max_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceClampMinTensor | ✅ | clamp_min_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceCopy | ✖️ |  | ✅ | ✅ | ✅ | Clone;Contiguous;Copy;IndexAddExt;InplaceCopy;MaskedFill;MaskedFillScalar;MaskedScatter;SilentCheckV2;SilentCheckV3;TensorScatterAdd;TypeAs |
| aclnnInplaceCos | ✅ | cos_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceCosh | ✅ | cosh_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceCumprod | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceDiv | ✅ | div_.Scalar;div_.Scalar_mode;div_.Tensor;div_.Tensor_mode | ✅ | ✅ | ✅ | InplaceDiv |
| aclnnInplaceDivMod | ✅ | div_.Scalar;div_.Scalar_mode;div_.Tensor;div_.Tensor_mode | ✅ | ✅ | ✅ | InplaceDivmod |
| aclnnInplaceDivMods | ✅ | div_.Scalar;div_.Scalar_mode;div_.Tensor;div_.Tensor_mode | ✅ | ✅ | ✅ | InplaceDivmods |
| aclnnInplaceDivs | ✅ | div_.Scalar;div_.Scalar_mode;div_.Tensor;div_.Tensor_mode | ✅ | ✅ | ✅ | InplaceDivs |
| aclnnInplaceElu | ✅ | elu_ | ✅ | ✅ | ✅ | InplaceElu |
| aclnnInplaceEqScalar | ✅ | eq_.Scalar;eq_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceEqTensor | ✅ | eq_.Scalar;eq_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceErf | ✅ | erf_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceErfc | ✅ | erfc_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceErfinv | ✅ | erfinv_ | ✅ | ✅ | ✅ | InplaceErfinv |
| aclnnInplaceExp | ✅ | exp_ | ✅ | ✅ | ✅ | InplaceExp |
| aclnnInplaceExp2 | ✅ | exp2_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceExpm1 | ✅ | expm1_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceFfnWorkerScheduler | ✅ | ffn_worker_scheduler;ffn_worker_scheduler_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceFillDiagonal | ✅ | fill_diagonal_ | ✅ | ✅ | ✅ | InplaceFillDiagonal |
| aclnnInplaceFillScalar | ✅ | fill_.Scalar;fill_.Tensor | ✅ | ✅ | ✅ | FillScalar;FillTensor;FullLike;InplaceFillScalar;InplaceFillTensor;NewFull;NewOnes |
| aclnnInplaceFillTensor | ✅ | fill_.Scalar;fill_.Tensor | ✅ | ✅ | ✅ | FillTensor;InplaceFillTensor |
| aclnnInplaceFloor | ✅ | floor_ | ✅ | ✅ | ✅ | InplaceFloor |
| aclnnInplaceFloorDivide | ✅ | floor_divide_.Scalar;floor_divide_.Tensor | ✅ | ✅ | ✅ | InplaceFloorDivide |
| aclnnInplaceFloorDivides | ✅ | floor_divide_.Scalar;floor_divide_.Tensor | ✅ | ✅ | ✅ | InplaceFloorDivides |
| aclnnInplaceFmodScalar | ✅ | fmod_.Scalar | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceFmodTensor | ✅ | fmod_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceFrac | ✅ | frac_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceGeScalar | ✅ | ge_.Scalar;ge_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceGeTensor | ✅ | ge_.Scalar;ge_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceGtScalar | ✅ | gt_.Scalar;gt_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceGtTensor | ✅ | gt_.Scalar;gt_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceHardsigmoid | ✅ | hardsigmoid_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceHardswish | ✅ | hardswish_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceHardtanh | ✅ | hardtanh_ | ✅ | ✅ | ✅ | InplaceHardtanh |
| aclnnInplaceIndexCopy | ✅ | index_copy_ | ✅ | ✅ | ✅ | InplaceIndexCopy |
| aclnnInplaceIndexFill | ✅ | index_fill_.int_Scalar;index_fill_.int_Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceIndexFillTensor | ✅ | index_fill_.int_Scalar;index_fill_.int_Tensor | ✅ | ✅ | ✅ | InplaceIndexFillScalar;InplaceIndexFillTensor |
| aclnnInplaceLeScalar | ✅ | le_.Scalar;le_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceLeTensor | ✅ | le_.Scalar;le_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceLeakyRelu | ✅ | leaky_relu_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceLerp | ✅ | lerp_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceLerps | ✅ | lerp_.Scalar | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceLog | ✅ | log_ | ✅ | ✅ | ✅ | InplaceLog |
| aclnnInplaceLog10 | ✅ | log10_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceLog1p | ✅ | log1p_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceLog2 | ✅ | log2_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceLogicalAnd | ✅ | logical_and_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceLogicalNot | ✅ | logical_not_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceLogicalOr | ✅ | logical_or_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceLtScalar | ✅ | lt_.Scalar;lt_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceLtTensor | ✅ | lt_.Scalar;lt_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceMaskedFillScalar | ✅ | masked_fill_.Scalar;masked_fill_.Tensor | ✅ | ✅ | ✅ | InplaceMaskedFillScalar;MaskedFillScalar |
| aclnnInplaceMaskedFillTensor | ✅ |  | ✅ | ✅ | ✅ | InplaceMaskedFillTensor;MaskedFill |
| aclnnInplaceMaskedScatter | ✅ |  | ✅ | ✅ | ✅ | InplaceMaskedScatter;MaskedScatter;MaskedSelectGrad |
| aclnnInplaceMatmulAllReduceAddRmsNorm | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceMish | ✅ | mish_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceMul | ✅ | mul_.Scalar;mul_.Tensor | ✅ | ✅ | ✅ | InplaceMul |
| aclnnInplaceMuls | ✅ | mul_.Scalar;mul_.Tensor | ✅ | ✅ | ✅ | InplaceMuls |
| aclnnInplaceNanToNum | ✅ | nan_to_num_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceNeScalar | ✅ | ne_.Scalar;ne_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceNeTensor | ✅ | ne_.Scalar;ne_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceNeg | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceNormal | ✅ | normal_ | ✅ | ✅ | ✅ | InplaceNormal;Randn;RandnLike |
| aclnnInplaceNormalTensor | ✅ | normal_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceOne | ✅ | one_;ones;ones.names;ones.out;ones_like | ✅ | ✅ | ✅ | Ones;OnesLikeExt |
| aclnnInplacePowTensorScalar | ✅ | pow_.Scalar | ✖️ | ✖️ | ✖️ |  |
| aclnnInplacePowTensorTensor | ✅ | pow_.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnInplacePut | ✅ |  | ✅ | ✅ | ✅ | InplacePut |
| aclnnInplaceQuantMatmulAllReduceAddRmsNorm | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceQuantScatter | ✅ | npu_quant_scatter;npu_quant_scatter_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceRReluWithNoise | ✅ | rrelu_with_noise_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceRandom | ✅ | random_;random_.from;random_.to | ✅ | ✅ | ✅ | InplaceRandom;RandInt;Randint;RandintLike |
| aclnnInplaceRandomTensor | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceReciprocal | ✅ | reciprocal_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceRelu | ✅ | relu_ | ✅ | ✅ | ✅ | InplaceReLU;InplaceRelu |
| aclnnInplaceRemainderTensorScalar | ✅ | remainder_.Scalar;remainder_.Tensor | ✅ | ✅ | ✅ | InplaceRemainderTensorScalar |
| aclnnInplaceRemainderTensorTensor | ✅ | remainder_.Scalar;remainder_.Tensor | ✅ | ✅ | ✅ | InplaceRemainderTensorTensor |
| aclnnInplaceRenorm | ✅ | renorm_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceRound | ✅ | round_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceRoundDecimals | ✅ | round_;round_.decimals | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceRsqrt | ✅ | rsqrt_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceScatter | ✅ | scatter_.src;scatter_.value | ✅ | ✅ | ✅ | InplaceScatterSrc;InplaceScatterSrcReduce |
| aclnnInplaceScatterUpdate | ✅ | scatter_update_ | ✅ | ✅ | ✅ | KVCacheScatterUpdate;KvCacheScatterUpdate |
| aclnnInplaceScatterValue | ✅ | scatter_.src;scatter_.value | ✅ | ✅ | ✅ | InplaceScatterValue;InplaceScatterValueReduce |
| aclnnInplaceSelu | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceSigmoid | ✅ | sigmoid_ | ✅ | ✅ | ✅ | InplaceSigmoid |
| aclnnInplaceSin | ✅ | sin_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceSinc | ✅ | sinc_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceSinh | ✅ | sinh_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceSqrt | ✅ | sqrt_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceSub | ✅ | sub_.Scalar;sub_.Tensor | ✅ | ✅ | ✖️ | InplaceSubExt |
| aclnnInplaceSubs | ✅ | sub_.Scalar;sub_.Tensor | ✅ | ✅ | ✅ | InplaceSubScalar |
| aclnnInplaceTan | ✅ | tan_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceTanh | ✅ | tanh_ | ✅ | ✅ | ✅ | InplaceTanh |
| aclnnInplaceThreshold | ✅ | threshold_ | ✅ | ✅ | ✅ | InplaceThreshold |
| aclnnInplaceTril | ✅ | tril_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceTriu | ✅ | triu_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceTrunc | ✅ | trunc_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceUniform | ✅ | uniform_ | ✅ | ✅ | ✅ | InplaceUniform;RandExt;RandLikeExt;UniformExt |
| aclnnInplaceUniformTensor | ✅ | uniform_ | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceXLogYScalarOther | ✅ | xlogy_.Scalar_Other | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceXLogYTensor | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnInplaceZero | ✅ | zero_;zeros;zeros.names;zeros.out | ✅ | ✅ | ✅ | GatherDGradV2;InplaceZero;IsInf;Isinf;MaskedSelectGrad;MultiScaleDeformableAttnGrad;NewZeros;Zeros;ZerosLikeExt |
| aclnnInstanceNorm | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnInverse | ✅ | inverse;inverse.out | ✅ | ✅ | ✅ | MatrixInverseExt |
| aclnnIou | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnIsClose | ✅ | isclose | ✅ | ✅ | ✅ | IsClose;Isclose |
| aclnnIsFinite | ✅ | isfinite | ✅ | ✅ | ✅ | IsFinite |
| aclnnIsInScalarTensor | ✅ | isin.Scalar_Tensor_out | ✖️ | ✖️ | ✖️ |  |
| aclnnIsInTensorScalar | ✅ | isin.Tensor_Scalar;isin.Tensor_Scalar_out | ✖️ | ✖️ | ✖️ |  |
| aclnnIsInf | ✖️ |  | ✅ | ✅ | ✅ | IsInf |
| aclnnIsNegInf | ✅ | isneginf.out | ✅ | ✅ | ✅ | IsNegInf |
| aclnnIsPosInf | ✅ | isposinf.out | ✖️ | ✖️ | ✖️ |  |
| aclnnKlDiv | ✅ | kl_div | ✅ | ✅ | ✅ | KLDiv;KlDiv |
| aclnnKlDivBackward | ✅ | kl_div_backward | ✅ | ✅ | ✅ | KLDivGrad;KlDivGrad |
| aclnnKthvalue | ✅ | kthvalue;kthvalue.dimname;kthvalue.dimname_out;kthvalue.values | ✅ | ✅ | ✅ | Kthvalue |
| aclnnL1Loss | ✅ | l1_loss | ✅ | ✅ | ✅ | L1LossExt |
| aclnnL1LossBackward | ✅ | l1_loss_backward | ✅ | ✅ | ✅ | L1LossBackwardExt |
| aclnnLayerNorm | ✅ |  | ✅ | ✅ | ✅ | LayerNormExt |
| aclnnLayerNormBackward | ✅ | native_layer_norm_backward | ✅ | ✅ | ✅ | LayerNormGradExt |
| aclnnLayerNormWithImplMode | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnLeScalar | ✅ | le.Scalar;le.Scalar_out;le.Tensor;le.Tensor_out | ✖️ | ✖️ | ✖️ |  |
| aclnnLeTensor | ✅ | le.Scalar;le.Scalar_out;le.Tensor;le.Tensor_out | ✅ | ✅ | ✅ | LessEqual |
| aclnnLeakyRelu | ✅ | leaky_relu;leaky_relu.out | ✅ | ✅ | ✅ | LeakyReLUExt |
| aclnnLeakyReluBackward | ✅ | leaky_relu_backward;leaky_relu_backward.grad_input | ✅ | ✅ | ✅ | LeakyReLUGradExt |
| aclnnLerp | ✅ | lerp.Tensor;lerp.Tensor_out | ✅ | ✅ | ✅ | Lerp |
| aclnnLerps | ✅ | lerp.Scalar;lerp.Scalar_out | ✅ | ✅ | ✅ | LerpScalar |
| aclnnLgamma | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnLightningIndexerGrad | ✖️ |  | ✅ | ✅ | ✅ | LightningIndexerGrad |
| aclnnLinalgCholesky | ✅ | linalg_cholesky;linalg_cholesky.out | ✖️ | ✖️ | ✖️ |  |
| aclnnLinalgCross | ✅ | linalg_cross;linalg_cross.out | ✅ | ✅ | ✅ | Cross |
| aclnnLinalgQr | ✅ | linalg_qr;linalg_qr.out | ✅ | ✅ | ✅ | LinalgQr |
| aclnnLinalgVectorNorm | ✅ | linalg_vector_norm;linalg_vector_norm.out | ✅ | ✅ | ✅ | LinalgVectorNorm |
| aclnnLinspace | ✅ | linspace;linspace.out | ✅ | ✅ | ✅ | LinSpaceExt |
| aclnnLog | ✅ | log;log.out | ✅ | ✅ | ✅ | Log |
| aclnnLog10 | ✅ | log10;log10.out | ✅ | ✅ | ✅ | Log10 |
| aclnnLog1p | ✅ | log1p;log1p.out | ✅ | ✅ | ✅ | Log1p |
| aclnnLog2 | ✅ | log2;log2.out | ✅ | ✅ | ✅ | Log2 |
| aclnnLogAddExp | ✅ | logaddexp;logaddexp.out | ✅ | ✅ | ✅ | LogAddExp |
| aclnnLogAddExp2 | ✅ | logaddexp2;logaddexp2.out | ✅ | ✅ | ✅ | LogAddExp2 |
| aclnnLogSigmoid | ✅ | log_sigmoid;log_sigmoid.out;log_sigmoid_forward;log_sigmoid_forward.output | ✖️ | ✖️ | ✖️ |  |
| aclnnLogSigmoidBackward | ✅ | log_sigmoid_backward;log_sigmoid_backward.grad_input | ✅ | ✅ | ✅ | LogSigmoidGrad |
| aclnnLogSigmoidForward | ✅ | log_sigmoid_forward;log_sigmoid_forward.output | ✅ | ✅ | ✅ | LogSigmoid |
| aclnnLogSoftmax | ✅ | _log_softmax | ✅ | ✅ | ✅ | LogSoftmax;LogSoftmaxExt |
| aclnnLogSoftmaxBackward | ✅ | _log_softmax_backward_data;_log_softmax_backward_data.out | ✅ | ✅ | ✅ | LogSoftmaxGrad |
| aclnnLogSumExp | ✅ | logsumexp;logsumexp.names;logsumexp.names_out;logsumexp.out | ✅ | ✅ | ✅ | LogSumExp |
| aclnnLogdet | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnLogicalAnd | ✅ | logical_and;logical_and.out | ✅ | ✅ | ✅ | LogicalAnd |
| aclnnLogicalNot | ✅ | logical_not;logical_not.out | ✅ | ✅ | ✅ | LogicalNot |
| aclnnLogicalOr | ✅ | logical_or;logical_or.out | ✅ | ✅ | ✅ | LogicalOr |
| aclnnLogicalXor | ✅ | logical_xor;logical_xor.out | ✅ | ✅ | ✅ | LogicalXor |
| aclnnLogit | ✅ | logit;logit.out | ✖️ | ✖️ | ✖️ |  |
| aclnnLogitGrad | ✅ | logit_backward;logit_backward.grad_input | ✖️ | ✖️ | ✖️ |  |
| aclnnLtScalar | ✅ | lt.Scalar;lt.Scalar_out;lt.Tensor;lt.Tensor_out | ✅ | ✅ | ✅ | LessScalar |
| aclnnLtTensor | ✅ | lt.Scalar;lt.Scalar_out;lt.Tensor;lt.Tensor_out | ✅ | ✅ | ✅ | Less |
| aclnnMaskedSelect | ✅ | masked_select;masked_select.out | ✅ | ✅ | ✅ | MaskedSelect |
| aclnnMaskedSoftmaxWithRelPosBias | ✅ | npu_masked_softmax_with_rel_pos_bias | ✖️ | ✖️ | ✖️ |  |
| aclnnMatmul | ✅ | npu_attn_softmax_backward_ | ✅ | ✅ | ✅ | BatchMatMul;Dense;MatMul;MatMulExt;MatMulV2;MatmulExt |
| aclnnMatmulAllReduce | ✅ | npu_mm_all_reduce_base | ✅ | ✖️ | ✅ | MatMulAllReduce |
| aclnnMatmulAllReduceAddRmsNorm | ✖️ |  | ✅ | ✅ | ✅ | MatmulAllReduceAddRmsNorm;MatmulAllreduceAddRmsnorm |
| aclnnMatmulAllReduceV2 | ✅ | npu_mm_all_reduce_base | ✖️ | ✖️ | ✖️ |  |
| aclnnMatmulCompress | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMatmulCompressDequant | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMatmulReduceScatter | ✅ |  | ✅ | ✅ | ✅ | MatmulReduceScatter |
| aclnnMatmulReduceScatterV2 | ✅ | npu_quant_mm_reduce_scatter | ✖️ | ✖️ | ✖️ |  |
| aclnnMatmulWeightNz | ✅ | mm;mm.out | ✖️ | ✖️ | ✖️ |  |
| aclnnMax | ✅ | max;max.dim;max.dim_max;max.names_dim;max.names_dim_max;max.out | ✅ | ✅ | ✅ | BincountExt;Max |
| aclnnMaxDim | ✅ | max;max.dim;max.dim_max;max.names_dim;max.names_dim_max;max.out | ✅ | ✅ | ✅ | ArgMaxWithValue;ArgmaxWithValue;MaxDim |
| aclnnMaxN | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMaxPool | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMaxPool2dWithIndices | ✅ | max_pool2d_with_indices;max_pool2d_with_indices.out | ✅ | ✅ | ✅ | MaxPoolWithIndices |
| aclnnMaxPool2dWithIndicesBackward | ✅ | max_pool2d_with_indices_backward;max_pool2d_with_indices_backward.grad_input | ✅ | ✅ | ✅ | MaxPoolGradWithIndices |
| aclnnMaxPool2dWithMask | ✅ | max_pool2d_with_indices;max_pool2d_with_indices.out | ✅ | ✅ | ✅ | MaxPoolWithMask |
| aclnnMaxPool2dWithMaskBackward | ✅ | max_pool2d_with_indices_backward;max_pool2d_with_indices_backward.grad_input | ✅ | ✅ | ✅ | MaxPoolGradWithMask |
| aclnnMaxPool3dWithArgmax | ✅ | max_pool3d_with_indices;max_pool3d_with_indices.out | ✖️ | ✖️ | ✖️ |  |
| aclnnMaxPool3dWithArgmaxBackward | ✅ | max_pool3d_with_indices_backward;max_pool3d_with_indices_backward.grad_input | ✖️ | ✖️ | ✖️ |  |
| aclnnMaxUnpool2d | ✅ | max_unpool2d;max_unpool2d.out | ✅ | ✅ | ✅ | MaxUnpool2DExt;MaxUnpool2dExt |
| aclnnMaxUnpool2dBackward | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMaxUnpool3d | ✅ | max_unpool3d;max_unpool3d.out | ✖️ | ✖️ | ✖️ |  |
| aclnnMaxUnpool3dBackward | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMaxV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMaximum | ✅ | max.out;maximum;maximum.out | ✅ | ✅ | ✅ | Maximum |
| aclnnMean | ✅ | mean;mean.dim;mean.names_dim;mean.names_out;mean.out | ✅ | ✅ | ✅ | AdaptiveAvgPool3DExt;AdaptiveAvgPool3dExt;MeanExt |
| aclnnMeanV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMedian | ✅ | median | ✅ | ✅ | ✅ | MedianExt |
| aclnnMedianDim | ✅ | median.dim;median.dim_values | ✅ | ✅ | ✅ | MedianDim |
| aclnnMin | ✅ | min;min.dim;min.dim_min;min.names_dim;min.names_dim_min;min.out | ✅ | ✅ | ✅ | BincountExt;Min |
| aclnnMinDim | ✅ | min;min.dim;min.dim_min;min.names_dim;min.names_dim_min;min.out | ✅ | ✅ | ✅ | ArgMinWithValue;ArgminWithValue;MinDim |
| aclnnMinN | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMinimum | ✅ | min.out;minimum.out | ✅ | ✅ | ✅ | Minimum |
| aclnnMish | ✅ | mish;mish.out | ✅ | ✅ | ✅ | MishExt |
| aclnnMishBackward | ✅ | mish_backward | ✅ | ✅ | ✅ | MishGradExt |
| aclnnMlaPreprocess | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMlaPreprocessV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMlaProlog | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMlaPrologV2WeightNz | ✅ | npu_mla_prolog_v2 | ✖️ | ✖️ | ✖️ |  |
| aclnnMlaPrologV3WeightNz | ✅ | npu_mla_prolog_v3;npu_mla_prolog_v3_functional | ✖️ | ✖️ | ✖️ |  |
| aclnnMm | ✅ | mm;mm.out;npu_linear;npu_linear_backward | ✅ | ✅ | ✅ | Matmul;Mm;MmExt |
| aclnnModulate | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnModulateBackward | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeComputeExpertTokens | ✅ | npu_moe_compute_expert_tokens | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeDistributeCombine | ✅ | npu_moe_distribute_combine | ✅ | ✅ | ✅ | MoeDistributeCombine |
| aclnnMoeDistributeCombineAddRmsNorm | ✅ | npu_moe_distribute_combine_add_rms_norm | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeDistributeCombineAddRmsNormV2 | ✅ | npu_moe_distribute_combine_add_rms_norm | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeDistributeCombineV2 | ✅ | npu_moe_distribute_combine_v2 | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeDistributeCombineV3 | ✅ | npu_moe_distribute_combine_v2 | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeDistributeCombineV4 | ✅ | npu_moe_distribute_combine_v2 | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeDistributeDispatch | ✅ | npu_moe_distribute_dispatch | ✅ | ✅ | ✅ | MoeDistributeDispatch |
| aclnnMoeDistributeDispatchV2 | ✅ | npu_moe_distribute_dispatch_v2 | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeDistributeDispatchV3 | ✅ | npu_moe_distribute_dispatch_v2 | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeDistributeDispatchV4 | ✅ | npu_moe_distribute_dispatch_v2 | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeFinalizeRouting | ✅ | npu_moe_finalize_routing | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeFinalizeRoutingV2 | ✅ | npu_moe_finalize_routing | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeFinalizeRoutingV2Grad | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeFusedTopk | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeGatingTopK | ✅ | npu_moe_gating_top_k | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeGatingTopKSoftmax | ✅ | npu_moe_gating_top_k_softmax | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeGatingTopKSoftmaxV2 | ✅ | npu_moe_gating_top_k_softmax_v2 | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeInitRouting | ✅ | npu_moe_init_routing | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeInitRoutingQuant | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeInitRoutingQuantV2 | ✅ | npu_moe_init_routing_quant | ✅ | ✖️ | ✅ | MoeInitRoutingQuantV2 |
| aclnnMoeInitRoutingV2 | ✅ | npu_moe_init_routing_v2 | ✅ | ✖️ | ✅ | MoeInitRoutingV2 |
| aclnnMoeInitRoutingV2Grad | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeInitRoutingV3 | ✅ | npu_moe_init_routing_v2 | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeTokenPermute | ✅ | npu_moe_token_permute | ✅ | ✅ | ✅ | MoeTokenPermute |
| aclnnMoeTokenPermuteGrad | ✅ | npu_moe_token_permute_grad | ✅ | ✅ | ✅ | MoeTokenPermuteGrad |
| aclnnMoeTokenPermuteWithEp | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeTokenPermuteWithEpGrad | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeTokenPermuteWithRoutingMap | ✅ | npu_moe_token_permute_with_routing_map | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeTokenPermuteWithRoutingMapGrad | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeTokenUnpermute | ✅ | npu_moe_token_unpermute | ✅ | ✅ | ✅ | InnerMoeTokenUnpermute |
| aclnnMoeTokenUnpermuteGrad | ✅ | npu_moe_token_unpermute_grad | ✅ | ✅ | ✅ | MoeTokenUnpermuteGrad |
| aclnnMoeTokenUnpermuteWithEp | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeTokenUnpermuteWithEpGrad | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeTokenUnpermuteWithRoutingMap | ✅ | _npu_moe_token_unpermute_with_routing_map | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeTokenUnpermuteWithRoutingMapGrad | ✅ | npu_moe_token_unpermute_with_routing_map_grad | ✖️ | ✖️ | ✖️ |  |
| aclnnMoeUpdateExpert | ✅ | npu_moe_update_expert | ✖️ | ✖️ | ✖️ |  |
| aclnnMrgbaCustom | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMseLoss | ✅ | mse_loss;mse_loss.out | ✅ | ✅ | ✅ | MSELossExt;MseLossExt |
| aclnnMseLossBackward | ✅ | mse_loss_backward;mse_loss_backward.grad_input | ✅ | ✅ | ✅ | MSELossGradExt;MseLossGradExt |
| aclnnMseLossOut | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMul | ✅ | mul.Scalar;mul.Tensor;mul.out | ✅ | ✅ | ✅ | Mul;Outer;Square |
| aclnnMuls | ✅ | mul.Scalar;mul.Tensor;mul.out | ✅ | ✅ | ✅ | Muls |
| aclnnMultiScaleDeformableAttentionGrad | ✖️ |  | ✅ | ✅ | ✅ | MultiScaleDeformableAttnGrad |
| aclnnMultiScaleDeformableAttnFunction | ✖️ |  | ✅ | ✅ | ✅ | MultiScaleDeformableAttn |
| aclnnMultilabelMarginLoss | ✅ | multilabel_margin_loss.out;multilabel_margin_loss_forward;multilabel_margin_loss_forward.output | ✖️ | ✖️ | ✖️ |  |
| aclnnMultinomial | ✅ | multinomial;multinomial.out | ✅ | ✅ | ✅ | MultinomialExt |
| aclnnMultinomialTensor | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnMv | ✅ | mv;mv.out | ✅ | ✅ | ✅ | Mv |
| aclnnNLLLoss | ✅ | nll_loss_forward;nll_loss_forward.output | ✅ | ✅ | ✅ | NLLLoss;Nllloss |
| aclnnNLLLoss2d | ✅ | nll_loss2d_forward;nll_loss2d_forward.output | ✅ | ✅ | ✅ | NLLLoss2d;Nllloss2d |
| aclnnNLLLoss2dBackward | ✅ | nll_loss2d_backward;nll_loss2d_backward.grad_input | ✅ | ✅ | ✅ | NLLLoss2dGrad;Nllloss2dGrad |
| aclnnNLLLossBackward | ✅ | nll_loss_backward;nll_loss_backward.grad_input | ✅ | ✅ | ✅ | NLLLossGrad;NlllossGrad |
| aclnnNanMedian | ✅ | nanmedian | ✖️ | ✖️ | ✖️ |  |
| aclnnNanMedianDim | ✅ | nanmedian.dim | ✖️ | ✖️ | ✖️ |  |
| aclnnNanToNum | ✅ | nan_to_num;nan_to_num.out | ✅ | ✅ | ✅ | NanToNum |
| aclnnNeScalar | ✅ | ne.Scalar;ne.Scalar_out;ne.Tensor;ne.Tensor_out | ✅ | ✅ | ✅ | CountNonZero;NeScalar |
| aclnnNeTensor | ✅ | ne.Scalar;ne.Scalar_out;ne.Tensor;ne.Tensor_out | ✅ | ✅ | ✅ | NotEqual |
| aclnnNeg | ✅ | neg;neg.out;neg_ | ✅ | ✅ | ✅ | Neg |
| aclnnNonMaxSuppression | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnNonzero | ✅ | nonzero;nonzero.out | ✅ | ✅ | ✅ | NonZero |
| aclnnNonzeroV2 | ✖️ |  | ✅ | ✅ | ✅ | InnerNonZero |
| aclnnNorm | ✅ | norm.Scalar;norm.ScalarOpt_dim;norm.ScalarOpt_dim_dtype;norm.ScalarOpt_dtype;norm.dtype_out;norm.out | ✅ | ✅ | ✅ | Norm |
| aclnnNormRopeConcat | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnNormRopeConcatBackward | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnNormalFloatFloat | ✅ | normal.Tensor_Tensor;normal.Tensor_Tensor_out;normal.Tensor_float;normal.Tensor_float_out;normal.float_Tensor;normal.float_Tensor_out;normal.float_float;normal.float_float_out | ✅ | ✅ | ✅ | NormalFloatFloat |
| aclnnNormalFloatTensor | ✅ | normal.Tensor_Tensor;normal.Tensor_Tensor_out;normal.Tensor_float;normal.Tensor_float_out;normal.float_Tensor;normal.float_Tensor_out;normal.float_float;normal.float_float_out | ✅ | ✅ | ✅ | NormalFloatTensor |
| aclnnNormalTensorFloat | ✅ | normal.Tensor_Tensor;normal.Tensor_Tensor_out;normal.Tensor_float;normal.Tensor_float_out;normal.float_Tensor;normal.float_Tensor_out;normal.float_float;normal.float_float_out | ✅ | ✅ | ✅ | NormalTensorFloat |
| aclnnNormalTensorTensor | ✅ | normal.Tensor_Tensor;normal.Tensor_Tensor_out;normal.Tensor_float;normal.Tensor_float_out;normal.float_Tensor;normal.float_Tensor_out;normal.float_float;normal.float_float_out | ✅ | ✅ | ✅ | NormalTensorTensor |
| aclnnNpuFormatCast | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnNsaCompress | ✅ | npu_nsa_compress | ✅ | ✅ | ✅ | NsaCompress |
| aclnnNsaCompressAttention | ✅ | npu_nsa_compress_attention | ✅ | ✅ | ✅ | NsaCompressAttention |
| aclnnNsaCompressAttentionInfer | ✅ | npu_nsa_compress_attention_infer | ✖️ | ✖️ | ✖️ |  |
| aclnnNsaCompressGrad | ✅ | npu_nsa_compress_grad | ✅ | ✅ | ✅ | NsaCompressGrad |
| aclnnNsaCompressWithCache | ✅ | npu_nsa_compress_infer.cache | ✖️ | ✖️ | ✖️ |  |
| aclnnNsaSelectedAttention | ✅ | npu_nsa_select_attention | ✅ | ✅ | ✅ | NsaSelectAttention |
| aclnnNsaSelectedAttentionGrad | ✅ | npu_nsa_select_attention_grad | ✅ | ✅ | ✅ | NsaSelectAttentionGrad |
| aclnnNsaSelectedAttentionInfer | ✅ | npu_nsa_select_attention_infer | ✖️ | ✖️ | ✖️ |  |
| aclnnObfuscationCalculate | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnObfuscationCalculateV2 | ✅ | obfuscation_calculate | ✖️ | ✖️ | ✖️ |  |
| aclnnObfuscationSetup | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnObfuscationSetupV2 | ✅ | obfuscation_finalize;obfuscation_initialize | ✖️ | ✖️ | ✖️ |  |
| aclnnOneHot | ✅ | npu_one_hot;one_hot | ✅ | ✅ | ✅ | OneHotExt |
| aclnnPdist | ✅ | _pdist_forward | ✖️ | ✖️ | ✖️ |  |
| aclnnPdistForward | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnPermute | ✖️ |  | ✅ | ✅ | ✅ | Dense;TExt;Transpose;TransposeExt;TransposeExtView |
| aclnnPolar | ✅ | polar;polar.out | ✅ | ✅ | ✅ | Polar |
| aclnnPowScalarTensor | ✅ | pow.Scalar;pow.Scalar_out | ✅ | ✅ | ✅ | PowScalarTensor |
| aclnnPowTensorScalar | ✅ | pow.Tensor_Scalar;pow.Tensor_Scalar_out | ✅ | ✅ | ✅ | PowTensorScalar |
| aclnnPowTensorTensor | ✅ | pow.Tensor_Tensor;pow.Tensor_Tensor_out | ✅ | ✅ | ✅ | Pow |
| aclnnPrecisionCompare | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnPrelu | ✅ | _prelu_kernel | ✅ | ✅ | ✅ | PReLU |
| aclnnPreluBackward | ✅ | _prelu_kernel_backward | ✅ | ✅ | ✅ | PReLUGrad |
| aclnnProd | ✅ | prod;prod.dim_int;prod.int_out | ✅ | ✅ | ✖️ | ProdExt |
| aclnnProdDim | ✅ | prod;prod.dim_int;prod.int_out | ✅ | ✅ | ✅ | ProdExt |
| aclnnPromptFlashAttention | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnPromptFlashAttentionV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnPromptFlashAttentionV3 | ✖️ |  | ✅ | ✅ | ✅ | PromptFlashAttention |
| aclnnQr | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnQuantConvolution | ✅ | npu_quant_conv2d | ✖️ | ✖️ | ✖️ |  |
| aclnnQuantGroupedMatmulDequant | ✅ | npu_quant_grouped_matmul_dequant | ✖️ | ✖️ | ✖️ |  |
| aclnnQuantMatmul | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnQuantMatmulAllReduce | ✅ | npu_mm_all_reduce_base | ✖️ | ✖️ | ✖️ |  |
| aclnnQuantMatmulAllReduceAddRmsNorm | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnQuantMatmulAllReduceV2 | ✅ | npu_mm_all_reduce_base | ✅ | ✖️ | ✅ | QuantBatchMatmulAllReduce |
| aclnnQuantMatmulAllReduceV3 | ✅ | npu_mm_all_reduce_base | ✖️ | ✖️ | ✖️ |  |
| aclnnQuantMatmulDequant | ✅ | npu_quant_matmul_dequant | ✖️ | ✖️ | ✖️ |  |
| aclnnQuantMatmulReduceSumWeightNz | ✅ | npu_quant_matmul_reduce_sum | ✖️ | ✖️ | ✖️ |  |
| aclnnQuantMatmulV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnQuantMatmulV3 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnQuantMatmulV4 | ✖️ |  | ✅ | ✅ | ✅ | QuantBatchMatmul |
| aclnnQuantMatmulV5 | ✅ | npu_quant_matmul | ✅ | ✅ | ✖️ | QuantMatmul |
| aclnnQuantMatmulWeightNz | ✅ | npu_quant_matmul | ✖️ | ✖️ | ✖️ |  |
| aclnnQuantize | ✅ | _quantize_per_channel_impl.out;_quantize_per_tensor_impl.out | ✖️ | ✖️ | ✖️ |  |
| aclnnQuantizedBatchNorm | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnRReluWithNoise | ✅ | rrelu_with_noise;rrelu_with_noise.out | ✖️ | ✖️ | ✖️ |  |
| aclnnRainFusionAttention | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnRandperm | ✅ | randperm;randperm.generator;randperm.generator_out;randperm.out | ✅ | ✅ | ✅ | RandpermExt |
| aclnnRange | ✅ | range;range.out;range.step | ✖️ | ✖️ | ✖️ |  |
| aclnnReal | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnReciprocal | ✅ | reciprocal;reciprocal.out | ✅ | ✅ | ✅ | Reciprocal |
| aclnnRecurrentGatedDeltaRule | ✅ | npu_recurrent_gated_delta_rule;npu_recurrent_gated_delta_rule_functional | ✖️ | ✖️ | ✖️ |  |
| aclnnReduceLogSum | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnReduceNansum | ✅ |  | ✅ | ✅ | ✅ | Nansum |
| aclnnReduceSum | ✅ | sum;sum.DimnameList_out;sum.IntList_out;sum.dim_DimnameList;sum.dim_IntList | ✅ | ✅ | ✅ | CountNonZero;ReduceSum;SumExt |
| aclnnReflectionPad1d | ✅ | reflection_pad1d;reflection_pad1d.out | ✅ | ✅ | ✅ | ReflectionPad1D |
| aclnnReflectionPad1dBackward | ✅ | reflection_pad1d_backward;reflection_pad1d_backward.grad_input | ✅ | ✅ | ✅ | ReflectionPad1DGrad |
| aclnnReflectionPad2d | ✅ | reflection_pad2d;reflection_pad2d.out | ✅ | ✅ | ✅ | ReflectionPad2D |
| aclnnReflectionPad2dBackward | ✅ | reflection_pad2d_backward;reflection_pad2d_backward.grad_input | ✅ | ✅ | ✅ | ReflectionPad2DGrad |
| aclnnReflectionPad3d | ✅ | reflection_pad3d;reflection_pad3d.out | ✅ | ✅ | ✅ | ReflectionPad3D |
| aclnnReflectionPad3dBackward | ✅ | reflection_pad3d_backward;reflection_pad3d_backward.grad_input | ✅ | ✅ | ✅ | ReflectionPad3DGrad |
| aclnnRelu | ✅ | relu | ✅ | ✅ | ✅ | ReLU |
| aclnnRemainderScalarTensor | ✅ | remainder.Scalar;remainder.Scalar_Tensor;remainder.Scalar_out;remainder.Tensor;remainder.Tensor_out | ✅ | ✅ | ✅ | RemainderScalarTensor |
| aclnnRemainderTensorScalar | ✅ | remainder.Scalar;remainder.Scalar_Tensor;remainder.Scalar_out;remainder.Tensor;remainder.Tensor_out | ✅ | ✅ | ✅ | RemainderTensorScalar |
| aclnnRemainderTensorTensor | ✅ | remainder.Scalar;remainder.Scalar_Tensor;remainder.Scalar_out;remainder.Tensor;remainder.Tensor_out | ✅ | ✅ | ✅ | RemainderTensorTensor |
| aclnnRenorm | ✅ | renorm;renorm.out | ✖️ | ✖️ | ✖️ |  |
| aclnnRepeat | ✅ | repeat | ✅ | ✅ | ✅ | Repeat;Tile |
| aclnnRepeatInterleave | ✅ | repeat_interleave.self_Tensor;repeat_interleave.self_int | ✅ | ✅ | ✖️ | RepeatInterleaveTensor |
| aclnnRepeatInterleaveInt | ✅ |  | ✅ | ✅ | ✅ | RepeatInterleaveInt |
| aclnnRepeatInterleaveIntWithDim | ✅ |  | ✅ | ✅ | ✅ | RepeatInterleaveInt |
| aclnnRepeatInterleaveTensor | ✖️ |  | ✅ | ✅ | ✅ |  |
| aclnnRepeatInterleaveWithDim | ✅ | repeat_interleave.self_Tensor;repeat_interleave.self_int | ✅ | ✅ | ✅ | RepeatInterleaveTensor |
| aclnnReplicationPad1d | ✅ | replication_pad1d;replication_pad1d.out | ✅ | ✅ | ✅ | ReplicationPad1D |
| aclnnReplicationPad1dBackward | ✅ | replication_pad1d_backward;replication_pad1d_backward.grad_input | ✅ | ✅ | ✅ | ReplicationPad1DGrad |
| aclnnReplicationPad2d | ✅ | replication_pad2d;replication_pad2d.out | ✅ | ✅ | ✅ | ReplicationPad2D |
| aclnnReplicationPad2dBackward | ✅ | replication_pad2d_backward;replication_pad2d_backward.grad_input | ✅ | ✅ | ✅ | ReplicationPad2DGrad |
| aclnnReplicationPad3d | ✅ | replication_pad3d;replication_pad3d.out | ✅ | ✅ | ✅ | ReplicationPad3D |
| aclnnReplicationPad3dBackward | ✅ | replication_pad3d_backward;replication_pad3d_backward.grad_input | ✅ | ✅ | ✅ | ReplicationPad3DGrad |
| aclnnResize | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnRightShift | ✅ | __irshift__.Scalar;__irshift__.Tensor;__rshift__.Scalar;__rshift__.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnRingAttentionUpdate | ✖️ |  | ✅ | ✅ | ✅ | RingAttentionUpdate |
| aclnnRingAttentionUpdateV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnRmsNorm | ✅ | npu_rms_norm | ✅ | ✅ | ✅ | RmsNorm |
| aclnnRmsNormGrad | ✅ | npu_rms_norm_backward | ✅ | ✅ | ✅ | RmsNormGrad |
| aclnnRmsNormQuant | ✅ | npu_rms_norm_quant | ✖️ | ✖️ | ✖️ |  |
| aclnnRoiAlign | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnRoiAlignV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnRoiAlignV2Backward | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnRoll | ✅ | roll | ✅ | ✅ | ✅ | Roll |
| aclnnRopeWithSinCosCache | ✅ | npu_mrope | ✖️ | ✖️ | ✖️ |  |
| aclnnRotaryPositionEmbedding | ✅ | npu_rotary_mul | ✅ | ✅ | ✅ | RotaryPositionEmbedding |
| aclnnRotaryPositionEmbeddingGrad | ✅ | npu_rotary_mul_backward | ✅ | ✅ | ✅ | RotaryPositionEmbeddingGrad |
| aclnnRound | ✅ | round;round.out | ✅ | ✅ | ✅ |  |
| aclnnRoundDecimals | ✅ | round;round.decimals;round.decimals_out;round.out | ✅ | ✅ | ✅ | Round |
| aclnnRsqrt | ✅ | rsqrt;rsqrt.out | ✅ | ✅ | ✅ | Rsqrt |
| aclnnRsub | ✅ | rsub.Tensor | ✖️ | ✖️ | ✖️ |  |
| aclnnRsubs | ✅ | rsub.Scalar | ✖️ | ✖️ | ✖️ |  |
| aclnnSWhere | ✅ | where;where.self;where.self_out | ✅ | ✅ | ✅ | Select |
| aclnnScale | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnScaledMaskedSoftmax | ✅ | _masked_softmax;npu_scaled_masked_softmax | ✖️ | ✖️ | ✖️ |  |
| aclnnScaledMaskedSoftmaxBackward | ✅ | npu_scaled_masked_softmax_backward | ✖️ | ✖️ | ✖️ |  |
| aclnnScatter | ✅ | scatter.src_out;scatter.value_out | ✅ | ✅ | ✅ | Scatter |
| aclnnScatterAdd | ✅ | scatter_add;scatter_add.dimname;scatter_add_ | ✅ | ✅ | ✅ | GatherDGradV2;InplaceScatterAdd;ScatterAddExt |
| aclnnScatterNd | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnScatterNdUpdate | ✅ | npu_scatter_nd_update;npu_scatter_nd_update_ | ✖️ | ✖️ | ✖️ |  |
| aclnnScatterPaKvCache | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnScatterValue | ✅ | scatter.src_out;scatter.value_out | ✅ | ✅ | ✅ | ScatterValue |
| aclnnSearchSorted | ✅ | searchsorted.Tensor;searchsorted.Tensor_out | ✅ | ✅ | ✅ | SearchSorted;Searchsorted |
| aclnnSearchSorteds | ✅ | searchsorted.Scalar | ✖️ | ✖️ | ✖️ |  |
| aclnnSelu | ✖️ |  | ✅ | ✅ | ✅ | SeLUExt |
| aclnnSeluBackward | ✖️ |  | ✅ | ✅ | ✅ | SeluGrad |
| aclnnShrink | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnSigmoid | ✅ | sigmoid;sigmoid.out | ✅ | ✅ | ✅ | Sigmoid |
| aclnnSigmoidBackward | ✅ | sigmoid_backward;sigmoid_backward.grad_input | ✅ | ✅ | ✅ | SigmoidGrad |
| aclnnSign | ✅ | sgn;sgn.out;sign;sign.out;sign_ | ✅ | ✅ | ✅ | InplaceSign;Sign |
| aclnnSignBitsPack | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnSignBitsUnpack | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnSignbit | ✅ | signbit.out | ✖️ | ✖️ | ✖️ |  |
| aclnnSilentCheck | ✅ | _npu_silent_check_v2 | ✅ | ✅ | ✅ | SilentCheckV2 |
| aclnnSilentCheckV2 | ✅ | _npu_silent_check_v3 | ✅ | ✅ | ✅ | SilentCheckV3 |
| aclnnSilu | ✅ | silu;silu.out;silu_ | ✅ | ✅ | ✅ | InplaceSiLU;InplaceSilu;SiLU |
| aclnnSiluBackward | ✅ | silu_backward;silu_backward.grad_input | ✅ | ✅ | ✅ | SiLUGrad |
| aclnnSimThreadExponential | ✅ | npu_sim_exponential_ | ✖️ | ✖️ | ✖️ |  |
| aclnnSin | ✅ | sin;sin.out | ✅ | ✅ | ✅ | Sin |
| aclnnSinc | ✅ | sinc;sinc.out | ✅ | ✅ | ✅ | Sinc |
| aclnnSinh | ✅ | sinh;sinh.out | ✅ | ✅ | ✅ | Sinh |
| aclnnSinkhorn | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnSlice | ✖️ |  | ✅ | ✅ | ✅ | Narrow;Slice;SliceExt |
| aclnnSliceV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnSlogdet | ✅ | slogdet | ✖️ | ✖️ | ✖️ |  |
| aclnnSmoothL1Loss | ✅ | smooth_l1_loss;smooth_l1_loss.out | ✅ | ✅ | ✅ | SmoothL1Loss |
| aclnnSmoothL1LossBackward | ✅ | smooth_l1_loss_backward;smooth_l1_loss_backward.grad_input | ✅ | ✅ | ✅ | SmoothL1LossGrad |
| aclnnSoftMarginLoss | ✅ | soft_margin_loss;soft_margin_loss.out | ✅ | ✅ | ✅ | SoftMarginLoss |
| aclnnSoftMarginLossBackward | ✅ | soft_margin_loss_backward;soft_margin_loss_backward.grad_input | ✅ | ✅ | ✅ | SoftMarginLossGrad |
| aclnnSoftmax | ✅ | _softmax;_softmax.out;npu_attn_softmax_ | ✅ | ✅ | ✅ | Softmax |
| aclnnSoftmaxBackward | ✅ | _softmax_backward_data;_softmax_backward_data.out;npu_attn_softmax_backward_ | ✅ | ✅ | ✅ | SoftmaxBackward |
| aclnnSoftmaxCrossEntropyWithLogits | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnSoftplus | ✅ | softplus;softplus.out | ✅ | ✅ | ✅ | SoftplusExt |
| aclnnSoftplusBackward | ✅ | softplus_backward.grad_input | ✅ | ✅ | ✅ | SoftplusGradExt |
| aclnnSoftshrink | ✅ | softshrink;softshrink.out | ✅ | ✅ | ✅ | SoftShrink;Softshrink |
| aclnnSoftshrinkBackward | ✅ | softshrink_backward;softshrink_backward.grad_input | ✅ | ✅ | ✅ | SoftShrinkGrad;SoftshrinkGrad |
| aclnnSort | ✅ | sort;sort.dimname;sort.dimname_values;sort.stable;sort.values;sort.values_stable | ✅ | ✅ | ✅ | ArgSort;SortExt |
| aclnnSparseFlashAttentionGrad | ✖️ |  | ✅ | ✅ | ✅ | SparseFlashAttentionGrad |
| aclnnSparseLightningIndexerGradKLLoss | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnSplitTensor | ✖️ |  | ✅ | ✅ | ✅ | Chunk;SplitTensor |
| aclnnSplitWithSize | ✅ | split_with_sizes_copy.out | ✅ | ✅ | ✅ | SplitWithSize |
| aclnnSqrt | ✅ | sqrt;sqrt.out | ✅ | ✅ | ✅ | Sqrt |
| aclnnSquaredRelu | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnStack | ✅ | stack;stack.out | ✅ | ✅ | ✅ | StackExt |
| aclnnStd | ✅ | std.correction;std.correction_out | ✅ | ✅ | ✅ | Std |
| aclnnStdMeanCorrection | ✅ | std_mean.correction | ✅ | ✅ | ✅ | StdMean |
| aclnnStridedSliceAssignV2 | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnSub | ✅ | sub.Scalar;sub.Tensor;sub.out | ✅ | ✅ | ✅ | Sub;SubExt |
| aclnnSubs | ✅ | sub.Scalar;sub.Tensor;sub.out | ✅ | ✅ | ✅ | SubScalar |
| aclnnSum | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnSvd | ✅ | _linalg_svd.U | ✖️ | ✖️ | ✖️ |  |
| aclnnSwiGlu | ✅ | npu_swiglu | ✅ | ✅ | ✅ | Swiglu |
| aclnnSwiGluGrad | ✅ | npu_swiglu_backward | ✅ | ✅ | ✅ | SwigluGrad |
| aclnnSwiGluQuant | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnSwiGluQuantV2 | ✅ | npu_swiglu_quant | ✖️ | ✖️ | ✖️ |  |
| aclnnSwinAttentionScoreQuant | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnSwinTransformerLnQkvQuant | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnSwish | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnSwishBackward | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnSyncBatchNormGatherStats | ✅ | batch_norm_gather_stats_update | ✖️ | ✖️ | ✖️ |  |
| aclnnTake | ✅ | take;take.out | ✅ | ✅ | ✅ | Take |
| aclnnTan | ✅ | tan;tan.out | ✅ | ✅ | ✅ | Tan |
| aclnnTanh | ✅ | tanh;tanh.out | ✅ | ✅ | ✅ | Tanh |
| aclnnTanhBackward | ✅ | tanh_backward;tanh_backward.grad_input | ✅ | ✅ | ✅ | TanhGrad |
| aclnnTfScatterAdd | ✖️ |  | ✅ | ✅ | ✅ | TensorScatterAdd |
| aclnnThreeInterpolateBackward | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnThreshold | ✅ | threshold;threshold.out | ✅ | ✅ | ✅ | Threshold |
| aclnnThresholdBackward | ✅ | threshold_backward | ✅ | ✅ | ✅ | ReluGrad;ThresholdGrad |
| aclnnTopKTopPSample | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnTopk | ✅ | topk;topk.values | ✅ | ✅ | ✅ | TopkExt |
| aclnnTrace | ✅ | trace | ✅ | ✅ | ✅ | TraceExt |
| aclnnTransConvolutionWeight | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnTransMatmulWeight | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnTransQuantParam | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnTransQuantParamV2 | ✅ | npu_trans_quant_param;npu_weight_quant_batchmatmul | ✅ | ✅ | ✖️ | QuantMatmul |
| aclnnTransQuantParamV3 | ✅ | npu_trans_quant_param | ✖️ | ✖️ | ✖️ |  |
| aclnnTransformBiasRescaleQkv | ✅ | _transform_bias_rescale_qkv | ✖️ | ✖️ | ✖️ |  |
| aclnnTransposeBatchMatMul | ✅ | npu_transpose_batchmatmul | ✖️ | ✖️ | ✖️ |  |
| aclnnTriangularSolve | ✅ | triangular_solve.X | ✅ | ✅ | ✅ | TriangularSolve |
| aclnnTril | ✅ | tril;tril.out | ✅ | ✅ | ✅ | TrilExt |
| aclnnTriu | ✅ | triu;triu.out | ✅ | ✅ | ✅ | Triu |
| aclnnTrunc | ✅ | trunc;trunc.out | ✅ | ✅ | ✅ | Trunc |
| aclnnUnfoldGrad | ✅ | unfold_backward | ✖️ | ✖️ | ✖️ |  |
| aclnnUnique | ✅ | _unique | ✅ | ✖️ | ✅ | InnerUnique |
| aclnnUnique2 | ✅ | _unique2 | ✅ | ✅ | ✅ | Unique2 |
| aclnnUniqueConsecutive | ✅ | unique_consecutive | ✅ | ✅ | ✅ | UniqueConsecutive |
| aclnnUniqueDim | ✅ | unique_dim | ✅ | ✅ | ✅ | UniqueDim |
| aclnnUpsampleBicubic2d | ✅ | upsample_bicubic2d;upsample_bicubic2d.out | ✅ | ✅ | ✅ | UpsampleBicubic2D;UpsampleBicubic2d |
| aclnnUpsampleBicubic2dAA | ✅ | _upsample_bicubic2d_aa;_upsample_bicubic2d_aa.out | ✖️ | ✖️ | ✖️ |  |
| aclnnUpsampleBicubic2dAAGrad | ✅ | _upsample_bicubic2d_aa_backward;_upsample_bicubic2d_aa_backward.grad_input | ✖️ | ✖️ | ✖️ |  |
| aclnnUpsampleBicubic2dBackward | ✅ | upsample_bicubic2d_backward;upsample_bicubic2d_backward.grad_input | ✅ | ✅ | ✅ | UpsampleBicubic2DGrad;UpsampleBicubic2dGrad |
| aclnnUpsampleBilinear2d | ✅ | upsample_bilinear2d;upsample_bilinear2d.out | ✅ | ✅ | ✅ | UpsampleBilinear2D;UpsampleBilinear2d |
| aclnnUpsampleBilinear2dAA | ✅ | _upsample_bilinear2d_aa;_upsample_bilinear2d_aa.out | ✖️ | ✖️ | ✖️ |  |
| aclnnUpsampleBilinear2dAABackward | ✅ | _upsample_bilinear2d_aa_backward;_upsample_bilinear2d_aa_backward.grad_input | ✖️ | ✖️ | ✖️ |  |
| aclnnUpsampleBilinear2dBackward | ✅ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnUpsampleBilinear2dBackwardV2 | ✅ | upsample_bilinear2d_backward;upsample_bilinear2d_backward.grad_input | ✅ | ✅ | ✅ | UpsampleBilinear2DGrad;UpsampleBilinear2dGrad |
| aclnnUpsampleLinear1d | ✅ | upsample_linear1d;upsample_linear1d.out | ✅ | ✅ | ✅ | UpsampleLinear1D;UpsampleLinear1d |
| aclnnUpsampleLinear1dBackward | ✅ | upsample_linear1d_backward | ✅ | ✅ | ✅ | UpsampleLinear1DGrad;UpsampleLinear1dGrad |
| aclnnUpsampleNearest1d | ✅ |  | ✅ | ✅ | ✅ | UpsampleNearest1D;UpsampleNearest1d |
| aclnnUpsampleNearest1dBackward | ✅ | upsample_nearest1d_backward;upsample_nearest1d_backward.grad_input | ✅ | ✅ | ✅ | UpsampleNearest1DGrad;UpsampleNearest1dGrad |
| aclnnUpsampleNearest1dV2 | ✅ | upsample_nearest1d;upsample_nearest1d.out | ✖️ | ✖️ | ✖️ |  |
| aclnnUpsampleNearest2d | ✅ |  | ✅ | ✅ | ✅ | UpsampleNearest2D;UpsampleNearest2d |
| aclnnUpsampleNearest2dBackward | ✅ | upsample_nearest2d_backward;upsample_nearest2d_backward.grad_input | ✅ | ✅ | ✅ | UpsampleNearest2DGrad;UpsampleNearest2dGrad |
| aclnnUpsampleNearest2dV2 | ✅ | upsample_nearest2d;upsample_nearest2d.out | ✖️ | ✖️ | ✖️ |  |
| aclnnUpsampleNearest3d | ✅ | upsample_nearest3d;upsample_nearest3d.out | ✅ | ✅ | ✅ | UpsampleNearest3D;UpsampleNearest3d |
| aclnnUpsampleNearest3dBackward | ✅ | upsample_nearest3d_backward;upsample_nearest3d_backward.grad_input | ✅ | ✅ | ✅ | UpsampleNearest3DGrad;UpsampleNearest3dGrad |
| aclnnUpsampleNearestExact1d | ✅ | _upsample_nearest_exact1d;_upsample_nearest_exact1d.out | ✖️ | ✖️ | ✖️ |  |
| aclnnUpsampleNearestExact1dBackward | ✅ | _upsample_nearest_exact1d_backward;_upsample_nearest_exact1d_backward.grad_input | ✖️ | ✖️ | ✖️ |  |
| aclnnUpsampleNearestExact2d | ✅ | _upsample_nearest_exact2d;_upsample_nearest_exact2d.out | ✖️ | ✖️ | ✖️ |  |
| aclnnUpsampleNearestExact2dBackward | ✅ | _upsample_nearest_exact2d_backward;_upsample_nearest_exact2d_backward.grad_input | ✖️ | ✖️ | ✖️ |  |
| aclnnUpsampleNearestExact3d | ✅ | _upsample_nearest_exact3d;_upsample_nearest_exact3d.out | ✖️ | ✖️ | ✖️ |  |
| aclnnUpsampleNearestExact3dBackward | ✅ | _upsample_nearest_exact3d_backward;_upsample_nearest_exact3d_backward.grad_input | ✖️ | ✖️ | ✖️ |  |
| aclnnUpsampleTrilinear3d | ✅ | upsample_trilinear3d;upsample_trilinear3d.out | ✅ | ✅ | ✅ | UpsampleTrilinear3D;UpsampleTrilinear3d |
| aclnnUpsampleTrilinear3dBackward | ✅ | upsample_trilinear3d_backward;upsample_trilinear3d_backward.grad_input | ✅ | ✅ | ✅ | UpsampleTrilinear3DGrad;UpsampleTrilinear3dGrad |
| aclnnVar | ✖️ |  | ✅ | ✅ | ✅ |  |
| aclnnVarCorrection | ✅ | var.correction;var.correction_out | ✅ | ✅ | ✅ | Var |
| aclnnVarMean | ✅ | var_mean.correction | ✅ | ✅ | ✅ | VarMean |
| aclnnWeightQuantBatchMatmul | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnWeightQuantBatchMatmulV2 | ✅ | npu_weight_quant_batchmatmul | ✅ | ✅ | ✅ | WeightQuantBatchMatmul |
| aclnnWeightQuantBatchMatmulV3 | ✅ | npu_weight_quant_batchmatmul | ✖️ | ✖️ | ✖️ |  |
| aclnnWeightQuantMatmulAllReduce | ✅ | npu_mm_all_reduce_base | ✖️ | ✖️ | ✖️ |  |
| aclnnWeightQuantMatmulAllReduceAddRmsNorm | ✖️ |  | ✖️ | ✖️ | ✖️ |  |
| aclnnXLogYScalarOther | ✅ | xlogy.OutScalar_Other;xlogy.Scalar_Other | ✅ | ✅ | ✅ | XLogYScalarOther |
| aclnnXLogYScalarSelf | ✅ | xlogy.OutScalar_Self;xlogy.Scalar_Self | ✅ | ✅ | ✅ | XLogYScalarSelf |
| aclnnXLogYTensor | ✅ | xlogy.OutTensor;xlogy.Tensor;xlogy_.Tensor | ✅ | ✅ | ✅ | Xlogy |
