# trt-summary

trt-summary provides information the summary of ONNX and Engine(Plan). (e.g., NetworkDefinition Layers List, Engine Layers List, or Binding Tensors Lists)

# Build

```
$ mkdir build && cd build
$ cmake ..
$ make
```

# Usage

```
./trtsummary --onnx onnx/file/path [text/file/path/for/export]
```
or
```
./trtsummary --engine plan/file/path [text/file/path/for/export]
```

Export file path is optional. If it is not given, using default stream.

# Example

## ONNX

- Onnx file for example: [resnet50-v1-12.onnx](https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-12.onnx)
```
./trtsummary --onnx resnet50-v1-12.onnx
[I] ----------------------------------------------------------------
[I] Filename : ../../resnet50-v1-12.onnx
[I] Format   : onnx
[I] ----------------------------------------------------------------
[I] [MemUsageChange] Init CUDA: CPU +329, GPU +0, now: CPU 337, GPU 1351 (MiB)
[I] [MemUsageChange] Init builder kernel library: CPU +327, GPU +104, now: CPU 683, GPU 1455 (MiB)
[I] ----------------------------------------------------------------
[I] Input filename:   ../../resnet50-v1-12.onnx
[I] ONNX IR version:  0.0.4
[I] Opset version:    12
[I] Producer name:    
[I] Producer version: 
[I] Domain:           
[I] Model version:    0
[I] Doc string:       
[I] ----------------------------------------------------------------
[W] onnx2trt_utils.cpp:369: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
[I] Parsing is done
> Engine Layer Summary:
----------------------------------------------------------------------------------------------------------
No   Layer Name                                        Type                     Output Shape        
==========================================================================================================
1    resnetv17_conv0_fwd                               Convolution              [-1, 64, 112, 112]        
2    resnetv17_batchnorm0_fwd                          Scale                    [-1, 64, 112, 112]        
3    resnetv17_relu0_fwd                               Activation               [-1, 64, 112, 112]        
4    resnetv17_pool0_fwd                               Pooling                  [-1, 64, 56, 56]          
5    resnetv17_stage1_conv0_fwd                        Convolution              [-1, 64, 56, 56]          
6    resnetv17_stage1_batchnorm0_fwd                   Scale                    [-1, 64, 56, 56]          
7    resnetv17_stage1_relu0_fwd                        Activation               [-1, 64, 56, 56]          
8    resnetv17_stage1_conv1_fwd                        Convolution              [-1, 64, 56, 56]          
9    resnetv17_stage1_batchnorm1_fwd                   Scale                    [-1, 64, 56, 56]          
10   resnetv17_stage1_relu1_fwd                        Activation               [-1, 64, 56, 56]          
11   resnetv17_stage1_conv2_fwd                        Convolution              [-1, 256, 56, 56]         
12   resnetv17_stage1_batchnorm2_fwd                   Scale                    [-1, 256, 56, 56]         
13   resnetv17_stage1_conv3_fwd                        Convolution              [-1, 256, 56, 56]         
14   resnetv17_stage1_batchnorm3_fwd                   Scale                    [-1, 256, 56, 56]         
15   resnetv17_stage1__plus0                           Elementwise              [-1, 256, 56, 56]         
16   resnetv17_stage1_activation0                      Activation               [-1, 256, 56, 56]         
17   resnetv17_stage1_conv4_fwd                        Convolution              [-1, 64, 56, 56]          
18   resnetv17_stage1_batchnorm4_fwd                   Scale                    [-1, 64, 56, 56]          
19   resnetv17_stage1_relu2_fwd                        Activation               [-1, 64, 56, 56]          
20   resnetv17_stage1_conv5_fwd                        Convolution              [-1, 64, 56, 56]          
21   resnetv17_stage1_batchnorm5_fwd                   Scale                    [-1, 64, 56, 56]          
22   resnetv17_stage1_relu3_fwd                        Activation               [-1, 64, 56, 56]          
23   resnetv17_stage1_conv6_fwd                        Convolution              [-1, 256, 56, 56]         
24   resnetv17_stage1_batchnorm6_fwd                   Scale                    [-1, 256, 56, 56]         
25   resnetv17_stage1__plus1                           Elementwise              [-1, 256, 56, 56]         
26   resnetv17_stage1_activation1                      Activation               [-1, 256, 56, 56]         
27   resnetv17_stage1_conv7_fwd                        Convolution              [-1, 64, 56, 56]          
28   resnetv17_stage1_batchnorm7_fwd                   Scale                    [-1, 64, 56, 56]          
29   resnetv17_stage1_relu4_fwd                        Activation               [-1, 64, 56, 56]          
30   resnetv17_stage1_conv8_fwd                        Convolution              [-1, 64, 56, 56]          
31   resnetv17_stage1_batchnorm8_fwd                   Scale                    [-1, 64, 56, 56]          
32   resnetv17_stage1_relu5_fwd                        Activation               [-1, 64, 56, 56]          
33   resnetv17_stage1_conv9_fwd                        Convolution              [-1, 256, 56, 56]         
34   resnetv17_stage1_batchnorm9_fwd                   Scale                    [-1, 256, 56, 56]         
35   resnetv17_stage1__plus2                           Elementwise              [-1, 256, 56, 56]         
36   resnetv17_stage1_activation2                      Activation               [-1, 256, 56, 56]         
37   resnetv17_stage2_conv0_fwd                        Convolution              [-1, 128, 28, 28]         
38   resnetv17_stage2_batchnorm0_fwd                   Scale                    [-1, 128, 28, 28]         
39   resnetv17_stage2_relu0_fwd                        Activation               [-1, 128, 28, 28]         
40   resnetv17_stage2_conv1_fwd                        Convolution              [-1, 128, 28, 28]         
41   resnetv17_stage2_batchnorm1_fwd                   Scale                    [-1, 128, 28, 28]         
42   resnetv17_stage2_relu1_fwd                        Activation               [-1, 128, 28, 28]         
43   resnetv17_stage2_conv2_fwd                        Convolution              [-1, 512, 28, 28]         
44   resnetv17_stage2_batchnorm2_fwd                   Scale                    [-1, 512, 28, 28]         
45   resnetv17_stage2_conv3_fwd                        Convolution              [-1, 512, 28, 28]         
46   resnetv17_stage2_batchnorm3_fwd                   Scale                    [-1, 512, 28, 28]         
47   resnetv17_stage2__plus0                           Elementwise              [-1, 512, 28, 28]         
48   resnetv17_stage2_activation0                      Activation               [-1, 512, 28, 28]         
49   resnetv17_stage2_conv4_fwd                        Convolution              [-1, 128, 28, 28]         
50   resnetv17_stage2_batchnorm4_fwd                   Scale                    [-1, 128, 28, 28]         
51   resnetv17_stage2_relu2_fwd                        Activation               [-1, 128, 28, 28]         
52   resnetv17_stage2_conv5_fwd                        Convolution              [-1, 128, 28, 28]         
53   resnetv17_stage2_batchnorm5_fwd                   Scale                    [-1, 128, 28, 28]         
54   resnetv17_stage2_relu3_fwd                        Activation               [-1, 128, 28, 28]         
55   resnetv17_stage2_conv6_fwd                        Convolution              [-1, 512, 28, 28]         
56   resnetv17_stage2_batchnorm6_fwd                   Scale                    [-1, 512, 28, 28]         
57   resnetv17_stage2__plus1                           Elementwise              [-1, 512, 28, 28]         
58   resnetv17_stage2_activation1                      Activation               [-1, 512, 28, 28]         
59   resnetv17_stage2_conv7_fwd                        Convolution              [-1, 128, 28, 28]         
60   resnetv17_stage2_batchnorm7_fwd                   Scale                    [-1, 128, 28, 28]         
61   resnetv17_stage2_relu4_fwd                        Activation               [-1, 128, 28, 28]         
62   resnetv17_stage2_conv8_fwd                        Convolution              [-1, 128, 28, 28]         
63   resnetv17_stage2_batchnorm8_fwd                   Scale                    [-1, 128, 28, 28]         
64   resnetv17_stage2_relu5_fwd                        Activation               [-1, 128, 28, 28]         
65   resnetv17_stage2_conv9_fwd                        Convolution              [-1, 512, 28, 28]         
66   resnetv17_stage2_batchnorm9_fwd                   Scale                    [-1, 512, 28, 28]         
67   resnetv17_stage2__plus2                           Elementwise              [-1, 512, 28, 28]         
68   resnetv17_stage2_activation2                      Activation               [-1, 512, 28, 28]         
69   resnetv17_stage2_conv10_fwd                       Convolution              [-1, 128, 28, 28]         
70   resnetv17_stage2_batchnorm10_fwd                  Scale                    [-1, 128, 28, 28]         
71   resnetv17_stage2_relu6_fwd                        Activation               [-1, 128, 28, 28]         
72   resnetv17_stage2_conv11_fwd                       Convolution              [-1, 128, 28, 28]         
73   resnetv17_stage2_batchnorm11_fwd                  Scale                    [-1, 128, 28, 28]         
74   resnetv17_stage2_relu7_fwd                        Activation               [-1, 128, 28, 28]         
75   resnetv17_stage2_conv12_fwd                       Convolution              [-1, 512, 28, 28]         
76   resnetv17_stage2_batchnorm12_fwd                  Scale                    [-1, 512, 28, 28]         
77   resnetv17_stage2__plus3                           Elementwise              [-1, 512, 28, 28]         
78   resnetv17_stage2_activation3                      Activation               [-1, 512, 28, 28]         
79   resnetv17_stage3_conv0_fwd                        Convolution              [-1, 256, 14, 14]         
80   resnetv17_stage3_batchnorm0_fwd                   Scale                    [-1, 256, 14, 14]         
81   resnetv17_stage3_relu0_fwd                        Activation               [-1, 256, 14, 14]         
82   resnetv17_stage3_conv1_fwd                        Convolution              [-1, 256, 14, 14]         
83   resnetv17_stage3_batchnorm1_fwd                   Scale                    [-1, 256, 14, 14]         
84   resnetv17_stage3_relu1_fwd                        Activation               [-1, 256, 14, 14]         
85   resnetv17_stage3_conv2_fwd                        Convolution              [-1, 1024, 14, 14]        
86   resnetv17_stage3_batchnorm2_fwd                   Scale                    [-1, 1024, 14, 14]        
87   resnetv17_stage3_conv3_fwd                        Convolution              [-1, 1024, 14, 14]        
88   resnetv17_stage3_batchnorm3_fwd                   Scale                    [-1, 1024, 14, 14]        
89   resnetv17_stage3__plus0                           Elementwise              [-1, 1024, 14, 14]        
90   resnetv17_stage3_activation0                      Activation               [-1, 1024, 14, 14]        
91   resnetv17_stage3_conv4_fwd                        Convolution              [-1, 256, 14, 14]         
92   resnetv17_stage3_batchnorm4_fwd                   Scale                    [-1, 256, 14, 14]         
93   resnetv17_stage3_relu2_fwd                        Activation               [-1, 256, 14, 14]         
94   resnetv17_stage3_conv5_fwd                        Convolution              [-1, 256, 14, 14]         
95   resnetv17_stage3_batchnorm5_fwd                   Scale                    [-1, 256, 14, 14]         
96   resnetv17_stage3_relu3_fwd                        Activation               [-1, 256, 14, 14]         
97   resnetv17_stage3_conv6_fwd                        Convolution              [-1, 1024, 14, 14]        
98   resnetv17_stage3_batchnorm6_fwd                   Scale                    [-1, 1024, 14, 14]        
99   resnetv17_stage3__plus1                           Elementwise              [-1, 1024, 14, 14]        
100  resnetv17_stage3_activation1                      Activation               [-1, 1024, 14, 14]        
101  resnetv17_stage3_conv7_fwd                        Convolution              [-1, 256, 14, 14]         
102  resnetv17_stage3_batchnorm7_fwd                   Scale                    [-1, 256, 14, 14]         
103  resnetv17_stage3_relu4_fwd                        Activation               [-1, 256, 14, 14]         
104  resnetv17_stage3_conv8_fwd                        Convolution              [-1, 256, 14, 14]         
105  resnetv17_stage3_batchnorm8_fwd                   Scale                    [-1, 256, 14, 14]         
106  resnetv17_stage3_relu5_fwd                        Activation               [-1, 256, 14, 14]         
107  resnetv17_stage3_conv9_fwd                        Convolution              [-1, 1024, 14, 14]        
108  resnetv17_stage3_batchnorm9_fwd                   Scale                    [-1, 1024, 14, 14]        
109  resnetv17_stage3__plus2                           Elementwise              [-1, 1024, 14, 14]        
110  resnetv17_stage3_activation2                      Activation               [-1, 1024, 14, 14]        
111  resnetv17_stage3_conv10_fwd                       Convolution              [-1, 256, 14, 14]         
112  resnetv17_stage3_batchnorm10_fwd                  Scale                    [-1, 256, 14, 14]         
113  resnetv17_stage3_relu6_fwd                        Activation               [-1, 256, 14, 14]         
114  resnetv17_stage3_conv11_fwd                       Convolution              [-1, 256, 14, 14]         
115  resnetv17_stage3_batchnorm11_fwd                  Scale                    [-1, 256, 14, 14]         
116  resnetv17_stage3_relu7_fwd                        Activation               [-1, 256, 14, 14]         
117  resnetv17_stage3_conv12_fwd                       Convolution              [-1, 1024, 14, 14]        
118  resnetv17_stage3_batchnorm12_fwd                  Scale                    [-1, 1024, 14, 14]        
119  resnetv17_stage3__plus3                           Elementwise              [-1, 1024, 14, 14]        
120  resnetv17_stage3_activation3                      Activation               [-1, 1024, 14, 14]        
121  resnetv17_stage3_conv13_fwd                       Convolution              [-1, 256, 14, 14]         
122  resnetv17_stage3_batchnorm13_fwd                  Scale                    [-1, 256, 14, 14]         
123  resnetv17_stage3_relu8_fwd                        Activation               [-1, 256, 14, 14]         
124  resnetv17_stage3_conv14_fwd                       Convolution              [-1, 256, 14, 14]         
125  resnetv17_stage3_batchnorm14_fwd                  Scale                    [-1, 256, 14, 14]         
126  resnetv17_stage3_relu9_fwd                        Activation               [-1, 256, 14, 14]         
127  resnetv17_stage3_conv15_fwd                       Convolution              [-1, 1024, 14, 14]        
128  resnetv17_stage3_batchnorm15_fwd                  Scale                    [-1, 1024, 14, 14]        
129  resnetv17_stage3__plus4                           Elementwise              [-1, 1024, 14, 14]        
130  resnetv17_stage3_activation4                      Activation               [-1, 1024, 14, 14]        
131  resnetv17_stage3_conv16_fwd                       Convolution              [-1, 256, 14, 14]         
132  resnetv17_stage3_batchnorm16_fwd                  Scale                    [-1, 256, 14, 14]         
133  resnetv17_stage3_relu10_fwd                       Activation               [-1, 256, 14, 14]         
134  resnetv17_stage3_conv17_fwd                       Convolution              [-1, 256, 14, 14]         
135  resnetv17_stage3_batchnorm17_fwd                  Scale                    [-1, 256, 14, 14]         
136  resnetv17_stage3_relu11_fwd                       Activation               [-1, 256, 14, 14]         
137  resnetv17_stage3_conv18_fwd                       Convolution              [-1, 1024, 14, 14]        
138  resnetv17_stage3_batchnorm18_fwd                  Scale                    [-1, 1024, 14, 14]        
139  resnetv17_stage3__plus5                           Elementwise              [-1, 1024, 14, 14]        
140  resnetv17_stage3_activation5                      Activation               [-1, 1024, 14, 14]        
141  resnetv17_stage4_conv0_fwd                        Convolution              [-1, 512, 7, 7]           
142  resnetv17_stage4_batchnorm0_fwd                   Scale                    [-1, 512, 7, 7]           
143  resnetv17_stage4_relu0_fwd                        Activation               [-1, 512, 7, 7]           
144  resnetv17_stage4_conv1_fwd                        Convolution              [-1, 512, 7, 7]           
145  resnetv17_stage4_batchnorm1_fwd                   Scale                    [-1, 512, 7, 7]           
146  resnetv17_stage4_relu1_fwd                        Activation               [-1, 512, 7, 7]           
147  resnetv17_stage4_conv2_fwd                        Convolution              [-1, 2048, 7, 7]          
148  resnetv17_stage4_batchnorm2_fwd                   Scale                    [-1, 2048, 7, 7]          
149  resnetv17_stage4_conv3_fwd                        Convolution              [-1, 2048, 7, 7]          
150  resnetv17_stage4_batchnorm3_fwd                   Scale                    [-1, 2048, 7, 7]          
151  resnetv17_stage4__plus0                           Elementwise              [-1, 2048, 7, 7]          
152  resnetv17_stage4_activation0                      Activation               [-1, 2048, 7, 7]          
153  resnetv17_stage4_conv4_fwd                        Convolution              [-1, 512, 7, 7]           
154  resnetv17_stage4_batchnorm4_fwd                   Scale                    [-1, 512, 7, 7]           
155  resnetv17_stage4_relu2_fwd                        Activation               [-1, 512, 7, 7]           
156  resnetv17_stage4_conv5_fwd                        Convolution              [-1, 512, 7, 7]           
157  resnetv17_stage4_batchnorm5_fwd                   Scale                    [-1, 512, 7, 7]           
158  resnetv17_stage4_relu3_fwd                        Activation               [-1, 512, 7, 7]           
159  resnetv17_stage4_conv6_fwd                        Convolution              [-1, 2048, 7, 7]          
160  resnetv17_stage4_batchnorm6_fwd                   Scale                    [-1, 2048, 7, 7]          
161  resnetv17_stage4__plus1                           Elementwise              [-1, 2048, 7, 7]          
162  resnetv17_stage4_activation1                      Activation               [-1, 2048, 7, 7]          
163  resnetv17_stage4_conv7_fwd                        Convolution              [-1, 512, 7, 7]           
164  resnetv17_stage4_batchnorm7_fwd                   Scale                    [-1, 512, 7, 7]           
165  resnetv17_stage4_relu4_fwd                        Activation               [-1, 512, 7, 7]           
166  resnetv17_stage4_conv8_fwd                        Convolution              [-1, 512, 7, 7]           
167  resnetv17_stage4_batchnorm8_fwd                   Scale                    [-1, 512, 7, 7]           
168  resnetv17_stage4_relu5_fwd                        Activation               [-1, 512, 7, 7]           
169  resnetv17_stage4_conv9_fwd                        Convolution              [-1, 2048, 7, 7]          
170  resnetv17_stage4_batchnorm9_fwd                   Scale                    [-1, 2048, 7, 7]          
171  resnetv17_stage4__plus2                           Elementwise              [-1, 2048, 7, 7]          
172  resnetv17_stage4_activation2                      Activation               [-1, 2048, 7, 7]          
173  resnetv17_pool1_fwd                               Reduce                   [-1, 2048, 1, 1]          
174  (Unnamed Layer* 173) [Constant]                   Constant                 [1]                       
175  (Unnamed Layer* 174) [Shape]                      Shape                    [4]                       
176  (Unnamed Layer* 175) [Gather]                     Gather                   [1]                       
177  (Unnamed Layer* 176) [Constant]                   Constant                 [1]                       
178  (Unnamed Layer* 177) [Concatenation]              Concatenation            [2]                       
179  flatten_473                                       Shuffle                  [-1, 2048]                
180  resnetv17_dense0_weight                           Constant                 [1000, 2048]              
181  resnetv17_dense0_fwd                              Matmul                   [-1, 1000]                
182  resnetv17_dense0_bias                             Constant                 [1000]                    
183  (Unnamed Layer* 182) [Shuffle]                    Shuffle                  [1, 1000]                 
184  (Unnamed Layer* 183) [ElementWise]                Elementwise              [-1, 1000]           (out)
----------------------------------------------------------------------------------------------------------
[W] Input 0 has multiple allowed formats: CHW | CHW2 | HWC8 | CHW4 | CHW16 | CHW32 | DHWC8 | CDHW32 | HWC | CHW(DLA) | HWC4(DLA) | HWC16
[W] Output 0 has multiple allowed formats: CHW | CHW2 | HWC8 | CHW4 | CHW16 | CHW32 | DHWC8 | CDHW32 | HWC | CHW(DLA) | HWC4(DLA) | HWC16
> Binding Tensor Summary:
--------------------------------------------------------------------------------------
Idx  Tensor Name                         Shape               Type      Format         
======================================================================================
0    data                          (in)  [-1, 3, 224, 224]   FP32      CHW            
0    resnetv17_dense0_fwd          (out) [-1, 1000]          FP32      CHW            
--------------------------------------------------------------------------------------
```

## PLAN

> Engine is compiled from [resnet50-v1-12.onnx](https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-12.onnx)

> Engine using custom plugin is not supported. If you a engine contained custom plugins, TensorRT library causes Segmentation Fault.

```
./trtsummary --engine resnet50-v1-12.plan
[I] -------------------------------------------------------------------------------------
[I] Filename : resnet50-v1-12-detailed.trt
[I] Format   : engine
[I] -------------------------------------------------------------------------------------
[I] [MemUsageChange] Init CUDA: CPU +329, GPU +0, now: CPU 435, GPU 1220 (MiB)
[I] Loaded engine size: 98 MiB
[I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +853, GPU +360, now: CPU 1309, GPU 1678 (MiB)
[I] [MemUsageChange] Init cuDNN: CPU +125, GPU +60, now: CPU 1434, GPU 1738 (MiB)
[I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +97, now: CPU 0, GPU 97 (MiB)
[I] Parsing is done
> Engine Layer Summary:
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
No   Layer Name                                                                                                              Type                          Output Shape        
===============================================================================================================================================================================
1    resnetv17_conv0_fwd + resnetv17_batchnorm0_fwd + resnetv17_relu0_fwd                                                    CaskConvolution               [1,64,112,112]      
2    resnetv17_pool0_fwd                                                                                                     TiledPooling                  [1,64,56,56]        
3    resnetv17_stage1_conv0_fwd + resnetv17_stage1_batchnorm0_fwd + resnetv17_stage1_relu0_fwd                               CaskConvolution               [1,64,56,56]        
4    resnetv17_stage1_conv1_fwd + resnetv17_stage1_batchnorm1_fwd + resnetv17_stage1_relu1_fwd                               CaskConvolution               [1,64,56,56]        
5    resnetv17_stage1_conv2_fwd + resnetv17_stage1_batchnorm2_fwd                                                            CaskConvolution               [1,256,56,56]       
6    resnetv17_stage1_conv3_fwd + resnetv17_stage1_batchnorm3_fwd + resnetv17_stage1__plus0 + resnetv17_stage1_activation0   CaskConvolution               [1,256,56,56]       
7    resnetv17_stage1_conv4_fwd + resnetv17_stage1_batchnorm4_fwd + resnetv17_stage1_relu2_fwd                               CaskConvolution               [1,64,56,56]        
8    resnetv17_stage1_conv5_fwd + resnetv17_stage1_batchnorm5_fwd + resnetv17_stage1_relu3_fwd                               CaskConvolution               [1,64,56,56]        
9    resnetv17_stage1_conv6_fwd + resnetv17_stage1_batchnorm6_fwd + resnetv17_stage1__plus1 + resnetv17_stage1_activation1   CaskConvolution               [1,256,56,56]       
10   resnetv17_stage1_conv7_fwd + resnetv17_stage1_batchnorm7_fwd + resnetv17_stage1_relu4_fwd                               CaskConvolution               [1,64,56,56]        
11   resnetv17_stage1_conv8_fwd + resnetv17_stage1_batchnorm8_fwd + resnetv17_stage1_relu5_fwd                               CaskConvolution               [1,64,56,56]        
12   resnetv17_stage1_conv9_fwd + resnetv17_stage1_batchnorm9_fwd + resnetv17_stage1__plus2 + resnetv17_stage1_activation2   CaskConvolution               [1,256,56,56]       
13   resnetv17_stage2_conv0_fwd + resnetv17_stage2_batchnorm0_fwd + resnetv17_stage2_relu0_fwd                               CaskConvolution               [1,128,28,28]       
14   resnetv17_stage2_conv1_fwd + resnetv17_stage2_batchnorm1_fwd + resnetv17_stage2_relu1_fwd                               FusedConvActConvolution       [1,128,28,28]       
15   resnetv17_stage2_conv2_fwd + resnetv17_stage2_batchnorm2_fwd                                                            CudnnConvolution              [1,512,28,28]       
16   resnetv17_stage2_conv3_fwd + resnetv17_stage2_batchnorm3_fwd + resnetv17_stage2__plus0 + resnetv17_stage2_activation0   CaskConvolution               [1,512,28,28]       
17   resnetv17_stage2_conv4_fwd + resnetv17_stage2_batchnorm4_fwd + resnetv17_stage2_relu2_fwd                               FusedConvActConvolution       [1,128,28,28]       
18   resnetv17_stage2_conv5_fwd + resnetv17_stage2_batchnorm5_fwd + resnetv17_stage2_relu3_fwd                               FusedConvActConvolution       [1,128,28,28]       
19   resnetv17_stage2_conv6_fwd + resnetv17_stage2_batchnorm6_fwd + resnetv17_stage2__plus1 + resnetv17_stage2_activation1   CaskConvolution               [1,512,28,28]       
20   resnetv17_stage2_conv7_fwd + resnetv17_stage2_batchnorm7_fwd + resnetv17_stage2_relu4_fwd                               FusedConvActConvolution       [1,128,28,28]       
21   resnetv17_stage2_conv8_fwd + resnetv17_stage2_batchnorm8_fwd + resnetv17_stage2_relu5_fwd                               FusedConvActConvolution       [1,128,28,28]       
22   resnetv17_stage2_conv9_fwd + resnetv17_stage2_batchnorm9_fwd + resnetv17_stage2__plus2 + resnetv17_stage2_activation2   CaskConvolution               [1,512,28,28]       
23   resnetv17_stage2_conv10_fwd + resnetv17_stage2_batchnorm10_fwd + resnetv17_stage2_relu6_fwd                             FusedConvActConvolution       [1,128,28,28]       
24   resnetv17_stage2_conv11_fwd + resnetv17_stage2_batchnorm11_fwd + resnetv17_stage2_relu7_fwd                             FusedConvActConvolution       [1,128,28,28]       
25   resnetv17_stage2_conv12_fwd + resnetv17_stage2_batchnorm12_fwd + resnetv17_stage2__plus3 + resnetv17_stage2_activation3 CaskConvolution               [1,512,28,28]       
26   resnetv17_stage3_conv0_fwd + resnetv17_stage3_batchnorm0_fwd + resnetv17_stage3_relu0_fwd                               CaskConvolution               [1,256,14,14]       
27   resnetv17_stage3_conv1_fwd + resnetv17_stage3_batchnorm1_fwd + resnetv17_stage3_relu1_fwd                               FusedConvActConvolution       [1,256,14,14]       
28   resnetv17_stage3_conv2_fwd + resnetv17_stage3_batchnorm2_fwd                                                            CudnnConvolution              [1,1024,14,14]      
29   resnetv17_stage3_conv3_fwd + resnetv17_stage3_batchnorm3_fwd + resnetv17_stage3__plus0 + resnetv17_stage3_activation0   CaskConvolution               [1,1024,14,14]      
30   resnetv17_stage3_conv4_fwd + resnetv17_stage3_batchnorm4_fwd + resnetv17_stage3_relu2_fwd                               FusedConvActConvolution       [1,256,14,14]       
31   resnetv17_stage3_conv5_fwd + resnetv17_stage3_batchnorm5_fwd + resnetv17_stage3_relu3_fwd                               FusedConvActConvolution       [1,256,14,14]       
32   resnetv17_stage3_conv6_fwd + resnetv17_stage3_batchnorm6_fwd + resnetv17_stage3__plus1 + resnetv17_stage3_activation1   CaskConvolution               [1,1024,14,14]      
33   resnetv17_stage3_conv7_fwd + resnetv17_stage3_batchnorm7_fwd + resnetv17_stage3_relu4_fwd                               FusedConvActConvolution       [1,256,14,14]       
34   resnetv17_stage3_conv8_fwd + resnetv17_stage3_batchnorm8_fwd + resnetv17_stage3_relu5_fwd                               FusedConvActConvolution       [1,256,14,14]       
35   resnetv17_stage3_conv9_fwd + resnetv17_stage3_batchnorm9_fwd + resnetv17_stage3__plus2 + resnetv17_stage3_activation2   CaskConvolution               [1,1024,14,14]      
36   resnetv17_stage3_conv10_fwd + resnetv17_stage3_batchnorm10_fwd + resnetv17_stage3_relu6_fwd                             FusedConvActConvolution       [1,256,14,14]       
37   resnetv17_stage3_conv11_fwd + resnetv17_stage3_batchnorm11_fwd + resnetv17_stage3_relu7_fwd                             FusedConvActConvolution       [1,256,14,14]       
38   resnetv17_stage3_conv12_fwd + resnetv17_stage3_batchnorm12_fwd + resnetv17_stage3__plus3 + resnetv17_stage3_activation3 CaskConvolution               [1,1024,14,14]      
39   resnetv17_stage3_conv13_fwd + resnetv17_stage3_batchnorm13_fwd + resnetv17_stage3_relu8_fwd                             FusedConvActConvolution       [1,256,14,14]       
40   resnetv17_stage3_conv14_fwd + resnetv17_stage3_batchnorm14_fwd + resnetv17_stage3_relu9_fwd                             FusedConvActConvolution       [1,256,14,14]       
41   resnetv17_stage3_conv15_fwd + resnetv17_stage3_batchnorm15_fwd + resnetv17_stage3__plus4 + resnetv17_stage3_activation4 CaskConvolution               [1,1024,14,14]      
42   resnetv17_stage3_conv16_fwd + resnetv17_stage3_batchnorm16_fwd + resnetv17_stage3_relu10_fwd                            FusedConvActConvolution       [1,256,14,14]       
43   resnetv17_stage3_conv17_fwd + resnetv17_stage3_batchnorm17_fwd + resnetv17_stage3_relu11_fwd                            FusedConvActConvolution       [1,256,14,14]       
44   resnetv17_stage3_conv18_fwd + resnetv17_stage3_batchnorm18_fwd + resnetv17_stage3__plus5 + resnetv17_stage3_activation5 CaskConvolution               [1,1024,14,14]      
45   Reformatting CopyNode for Output Tensor 0 to resnetv17_stage3_conv18_fwd + resnetv17_stage3_batchnorm18_fwd + resnet... Reformat                      [1,1024,14,14]      
46   resnetv17_stage4_conv0_fwd + resnetv17_stage4_batchnorm0_fwd + resnetv17_stage4_relu0_fwd                               CaskConvolution               [1,512,7,7]         
47   Reformatting CopyNode for Input Tensor 0 to resnetv17_stage4_conv1_fwd + resnetv17_stage4_batchnorm1_fwd + resnetv17... Reformat                      [1,512,7,7]         
48   resnetv17_stage4_conv1_fwd + resnetv17_stage4_batchnorm1_fwd + resnetv17_stage4_relu1_fwd                               FusedConvActConvolution       [1,512,7,7]         
49   resnetv17_stage4_conv2_fwd + resnetv17_stage4_batchnorm2_fwd                                                            FusedConvActConvolution       [1,2048,7,7]        
50   Reformatting CopyNode for Input Tensor 1 to resnetv17_stage4_conv3_fwd + resnetv17_stage4_batchnorm3_fwd + resnetv17... Reformat                      [1,2048,7,7]        
51   resnetv17_stage4_conv3_fwd + resnetv17_stage4_batchnorm3_fwd + resnetv17_stage4__plus0 + resnetv17_stage4_activation0   CaskConvolution               [1,2048,7,7]        
52   Reformatting CopyNode for Output Tensor 0 to resnetv17_stage4_conv3_fwd + resnetv17_stage4_batchnorm3_fwd + resnetv1... Reformat                      [1,2048,7,7]        
53   resnetv17_stage4_conv4_fwd + resnetv17_stage4_batchnorm4_fwd + resnetv17_stage4_relu2_fwd                               CudnnConvolution              [1,512,7,7]         
54   resnetv17_stage4_conv5_fwd + resnetv17_stage4_batchnorm5_fwd + resnetv17_stage4_relu3_fwd                               FusedConvActConvolution       [1,512,7,7]         
55   resnetv17_stage4_conv6_fwd + resnetv17_stage4_batchnorm6_fwd + resnetv17_stage4__plus1 + resnetv17_stage4_activation1   CudnnConvolution              [1,2048,7,7]        
56   resnetv17_stage4_conv7_fwd + resnetv17_stage4_batchnorm7_fwd + resnetv17_stage4_relu4_fwd                               CudnnConvolution              [1,512,7,7]         
57   resnetv17_stage4_conv8_fwd + resnetv17_stage4_batchnorm8_fwd + resnetv17_stage4_relu5_fwd                               FusedConvActConvolution       [1,512,7,7]         
58   resnetv17_stage4_conv9_fwd + resnetv17_stage4_batchnorm9_fwd + resnetv17_stage4__plus2 + resnetv17_stage4_activation2   CudnnConvolution              [1,2048,7,7]        
59   resnetv17_pool1_fwd                                                                                                     CudnnPooling                  [1,2048,1,1]        
60   resnetv17_dense0_fwd                                                                                                    CublasConvolution             [1,1000,1,1]        
61   reshape_after_resnetv17_dense0_fwd                                                                                      NoOp                          [1,1000]            
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
> Engine Binding Tensor Summary:
--------------------------------------------------------------------------------------
Idx  Tensor Name                         Shape               Type      Format         
======================================================================================
0    data                          (in)  [1, 3, 224, 224]    FP32      CHW            
1    resnetv17_dense0_fwd          (out) [1, 1000]           FP32      CHW            
--------------------------------------------------------------------------------------
```