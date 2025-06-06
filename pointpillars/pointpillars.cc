/******************************************************************************
 * Copyright 2020 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @author Kosuke Murakami
 * @date 2019/02/26
 */

/**
* @author Yan haixu
* Contact: just github.com/hova88
* @date 2021/04/30
*/


#include "pointpillars.h"

#include <chrono>
#include <iostream>


void PointPillars::InitParams()
{
    YAML::Node params = YAML::LoadFile(pp_config_);
    // 读取体素尺寸
    kPillarXSize = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][0].as<float>();
    kPillarYSize = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][1].as<float>();
    kPillarZSize = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][2].as<float>();
    // 读取点云范围
    kMinXRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][0].as<float>();
    kMinYRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][1].as<float>();
    kMinZRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][2].as<float>();
    kMaxXRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][3].as<float>();
    kMaxYRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][4].as<float>();
    kMaxZRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][5].as<float>();
    // 读取类别数、最大体素数
    kNumClass = params["CLASS_NAMES"].size();
    kMaxNumPillars = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["MAX_NUMBER_OF_VOXELS"]["test"].as<int>(); // 30000
    kMaxNumPointsPerPillar = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["MAX_POINTS_PER_VOXEL"].as<int>(); // 20
    kNumPointFeature = 5; // [x, y, z, i,0]
    kNumInputBoxFeature = 7;
    kNumOutputBoxFeature = params["MODEL"]["DENSE_HEAD"]["TARGET_ASSIGNER_CONFIG"]["BOX_CODER_CONFIG"]["code_size"].as<int>();//9
    kBatchSize = 1;
    kNumIndsForScan = 1024;
    kNumThreads = 64;
    kNumBoxCorners = 8;
    // 表示每隔4个单元格放置一个锚框
    kAnchorStrides = 4;
    kNmsPreMaxsize = params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["NMS_PRE_MAXSIZE"].as<int>();
    kNmsPostMaxsize = params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["NMS_POST_MAXSIZE"].as<int>();
    //params for initialize anchors
    //Adapt to OpenPCDet
    // 读取锚框尺寸和底部高度（适应 OpenPCDet）
    kAnchorNames = params["CLASS_NAMES"].as<std::vector<std::string>>();
    for (int i = 0; i < kAnchorNames.size(); ++i)
    {
        kAnchorDxSizes.emplace_back(params["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"][i]["anchor_sizes"][0][0].as<float>());
        kAnchorDySizes.emplace_back(params["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"][i]["anchor_sizes"][0][1].as<float>());
        kAnchorDzSizes.emplace_back(params["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"][i]["anchor_sizes"][0][2].as<float>());
        kAnchorBottom.emplace_back(params["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"][i]["anchor_bottom_heights"][0].as<float>());
    }
    for (int idx_head = 0; idx_head < params["MODEL"]["DENSE_HEAD"]["RPN_HEAD_CFGS"].size(); ++idx_head)
    {
        int num_cls_per_head = params["MODEL"]["DENSE_HEAD"]["RPN_HEAD_CFGS"][idx_head]["HEAD_CLS_NAME"].size();
        std::vector<int> value;
        for (int i = 0; i < num_cls_per_head; ++i)
        {
            value.emplace_back(idx_head + i);
        }
        kMultiheadLabelMapping.emplace_back(value);
    }
    // 计算网格尺寸和锚框数量
    // Generate secondary parameters based on above.
    kGridXSize = static_cast<int>((kMaxXRange - kMinXRange) / kPillarXSize); //512
    kGridYSize = static_cast<int>((kMaxYRange - kMinYRange) / kPillarYSize); //512
    kGridZSize = static_cast<int>((kMaxZRange - kMinZRange) / kPillarZSize); //1
    kRpnInputSize = 64 * kGridYSize * kGridXSize;  //64是特征维度

    kNumAnchorXinds = static_cast<int>(kGridXSize / kAnchorStrides); //Width
    kNumAnchorYinds = static_cast<int>(kGridYSize / kAnchorStrides); //Hight
    kNumAnchor = kNumAnchorXinds * kNumAnchorYinds * 2 * kNumClass;  // H * W * Ro * N = 196608

    kNumAnchorPerCls = kNumAnchorXinds * kNumAnchorYinds * 2; //H * W * Ro = 32768
    kRpnBoxOutputSize = kNumAnchor * kNumOutputBoxFeature; //边界框输出尺寸
    kRpnClsOutputSize = kNumAnchor * kNumClass; //类别分数输出尺寸
    kRpnDirOutputSize = kNumAnchor * 2; //方向分类输出尺寸
}


PointPillars::PointPillars(const float score_threshold,
                           const float nms_overlap_threshold,
                           const bool use_onnx,
                           const std::string pfe_file,
                           const std::string backbone_file,
                           const std::string pp_config)
    : score_threshold_(score_threshold),
      nms_overlap_threshold_(nms_overlap_threshold),
      use_onnx_(use_onnx),
      pfe_file_(pfe_file),
      backbone_file_(backbone_file),
      pp_config_(pp_config)
{
    InitParams();
    InitTRT(use_onnx_);   // 初始化 TensorRT 引擎
    DeviceMemoryMalloc(); // 分配GPU内存
    
    // 实例化 CUDA 处理模块
    preprocess_points_cuda_ptr_.reset(new PreprocessPointsCuda(
        kNumThreads,
        kMaxNumPillars,
        kMaxNumPointsPerPillar,
        kNumPointFeature,
        kNumIndsForScan,
        kGridXSize,kGridYSize, kGridZSize,
        kPillarXSize,kPillarYSize, kPillarZSize,
        kMinXRange, kMinYRange, kMinZRange));

    scatter_cuda_ptr_.reset(new ScatterCuda(kNumThreads, kGridXSize, kGridYSize));

    const float float_min = std::numeric_limits<float>::lowest();
    const float float_max = std::numeric_limits<float>::max();
    postprocess_cuda_ptr_.reset(
      new PostprocessCuda(kNumThreads,
                          float_min, float_max, 
                          kNumClass,kNumAnchorPerCls,
                          kMultiheadLabelMapping,
                          score_threshold_, 
                          nms_overlap_threshold_,
                          kNmsPreMaxsize, 
                          kNmsPostMaxsize,
                          kNumBoxCorners, 
                          kNumInputBoxFeature,
                          kNumOutputBoxFeature));  /*kNumOutputBoxFeature*/
    
}

// GPU内存分配与释放
void PointPillars::DeviceMemoryMalloc() {
    // for pillars 
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_num_points_per_pillar_), kMaxNumPillars * sizeof(float))); // M
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_x_coors_), kMaxNumPillars * sizeof(int))); // M
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_y_coors_), kMaxNumPillars * sizeof(int))); // M
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_pillar_point_feature_), kMaxNumPillars * kMaxNumPointsPerPillar * kNumPointFeature * sizeof(float))); // [M , m , 4]
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_pillar_coors_),  kMaxNumPillars * 4 * sizeof(float))); // [M , 4]
    // for sparse map
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_sparse_pillar_map_), kNumIndsForScan * kNumIndsForScan * sizeof(int))); // [1024 , 1024]
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_cumsum_along_x_), kNumIndsForScan * kNumIndsForScan * sizeof(int))); // [1024 , 1024]
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_cumsum_along_y_), kNumIndsForScan * kNumIndsForScan * sizeof(int)));// [1024 , 1024]

    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_pfe_gather_feature_),
                        kMaxNumPillars * kMaxNumPointsPerPillar *
                            kNumGatherPointFeature * sizeof(float)));
    // for trt inference
    // create GPU buffers and a stream

    GPU_CHECK(
        cudaMalloc(&pfe_buffers_[0], kMaxNumPillars * kMaxNumPointsPerPillar *
                                        kNumGatherPointFeature * sizeof(float)));
    GPU_CHECK(cudaMalloc(&pfe_buffers_[1], kMaxNumPillars * 64 * sizeof(float)));

    GPU_CHECK(cudaMalloc(&rpn_buffers_[0],  kRpnInputSize * sizeof(float)));

    GPU_CHECK(cudaMalloc(&rpn_buffers_[1],  kNumAnchorPerCls  * sizeof(float)));  //classes
    GPU_CHECK(cudaMalloc(&rpn_buffers_[2],  kNumAnchorPerCls  * 2 * 2 * sizeof(float)));
    GPU_CHECK(cudaMalloc(&rpn_buffers_[3],  kNumAnchorPerCls  * 2 * 2 * sizeof(float)));
    GPU_CHECK(cudaMalloc(&rpn_buffers_[4],  kNumAnchorPerCls  * sizeof(float)));
    GPU_CHECK(cudaMalloc(&rpn_buffers_[5],  kNumAnchorPerCls  * 2 * 2 * sizeof(float)));
    GPU_CHECK(cudaMalloc(&rpn_buffers_[6],  kNumAnchorPerCls  * 2 * 2 * sizeof(float)));
    
    GPU_CHECK(cudaMalloc(&rpn_buffers_[7],  kNumAnchorPerCls * kNumClass * kNumOutputBoxFeature * sizeof(float))); //boxes

    // for scatter kernel
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_scattered_feature_),
                        kNumThreads * kGridYSize * kGridXSize * sizeof(float)));

    // for filter
    host_box_ =  new float[kNumAnchorPerCls * kNumClass * kNumOutputBoxFeature]();
    host_score_ =  new float[kNumAnchorPerCls * 18]();
    host_filtered_count_ = new int[kNumClass]();
}


PointPillars::~PointPillars() {
    // for pillars 
    GPU_CHECK(cudaFree(dev_num_points_per_pillar_));
    GPU_CHECK(cudaFree(dev_x_coors_));
    GPU_CHECK(cudaFree(dev_y_coors_));
    GPU_CHECK(cudaFree(dev_pillar_point_feature_));
    GPU_CHECK(cudaFree(dev_pillar_coors_));
    // for sparse map
    GPU_CHECK(cudaFree(dev_sparse_pillar_map_));    
    GPU_CHECK(cudaFree(dev_cumsum_along_x_));
    GPU_CHECK(cudaFree(dev_cumsum_along_y_));
    // for pfe forward
    GPU_CHECK(cudaFree(dev_pfe_gather_feature_));
      
    GPU_CHECK(cudaFree(pfe_buffers_[0]));
    GPU_CHECK(cudaFree(pfe_buffers_[1]));

    GPU_CHECK(cudaFree(rpn_buffers_[0]));
    GPU_CHECK(cudaFree(rpn_buffers_[1]));
    GPU_CHECK(cudaFree(rpn_buffers_[2]));
    GPU_CHECK(cudaFree(rpn_buffers_[3]));
    GPU_CHECK(cudaFree(rpn_buffers_[4]));
    GPU_CHECK(cudaFree(rpn_buffers_[5]));
    GPU_CHECK(cudaFree(rpn_buffers_[6]));
    GPU_CHECK(cudaFree(rpn_buffers_[7]));
    pfe_context_->destroy();
    backbone_context_->destroy();
    pfe_engine_->destroy();
    backbone_engine_->destroy();
    // for post process
    GPU_CHECK(cudaFree(dev_scattered_feature_));
    delete[] host_box_;
    delete[] host_score_;
    delete[] host_filtered_count_;

}


// 清空GPU内存
void PointPillars::SetDeviceMemoryToZero() {

    GPU_CHECK(cudaMemset(dev_num_points_per_pillar_, 0, kMaxNumPillars * sizeof(float)));
    GPU_CHECK(cudaMemset(dev_x_coors_,               0, kMaxNumPillars * sizeof(int)));
    GPU_CHECK(cudaMemset(dev_y_coors_,               0, kMaxNumPillars * sizeof(int)));
    GPU_CHECK(cudaMemset(dev_pillar_point_feature_,  0, kMaxNumPillars * kMaxNumPointsPerPillar * kNumPointFeature * sizeof(float)));
    GPU_CHECK(cudaMemset(dev_pillar_coors_,          0, kMaxNumPillars * 4 * sizeof(float)));
    GPU_CHECK(cudaMemset(dev_sparse_pillar_map_,     0, kNumIndsForScan * kNumIndsForScan * sizeof(int)));

    GPU_CHECK(cudaMemset(dev_pfe_gather_feature_,    0, kMaxNumPillars * kMaxNumPointsPerPillar * kNumGatherPointFeature * sizeof(float)));
    GPU_CHECK(cudaMemset(pfe_buffers_[0],       0, kMaxNumPillars * kMaxNumPointsPerPillar * kNumGatherPointFeature * sizeof(float)));
    GPU_CHECK(cudaMemset(pfe_buffers_[1],       0, kMaxNumPillars * 64 * sizeof(float)));

    GPU_CHECK(cudaMemset(dev_scattered_feature_,    0, kNumThreads * kGridYSize * kGridXSize * sizeof(float)));
}





void PointPillars::InitTRT(const bool use_onnx) {
  if (use_onnx_) {
    // create a TensorRT model from the onnx model and load it into an engine
    OnnxToTRTModel(pfe_file_, &pfe_engine_);
    OnnxToTRTModel(backbone_file_, &backbone_engine_);
  }else {
    EngineToTRTModel(pfe_file_, &pfe_engine_);
    EngineToTRTModel(backbone_file_, &backbone_engine_);
  }
    if (pfe_engine_ == nullptr || backbone_engine_ == nullptr) {
        std::cerr << "Failed to load ONNX file.";
    }

    // create execution context from the engine
    pfe_context_ = pfe_engine_->createExecutionContext();
    backbone_context_ = backbone_engine_->createExecutionContext();
    if (pfe_context_ == nullptr || backbone_context_ == nullptr) {
        std::cerr << "Failed to create TensorRT Execution Context.";
    }
  
}

void PointPillars::OnnxToTRTModel(
    const std::string& model_file,  // name of the onnx model
    nvinfer1::ICudaEngine** engine_ptr) {
    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);

    // create the builder
    const auto explicit_batch =
        static_cast<uint32_t>(kBatchSize) << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(g_logger_);
    nvinfer1::INetworkDefinition* network =
        builder->createNetworkV2(explicit_batch);

    // parse onnx model
    auto parser = nvonnxparser::createParser(*network, g_logger_);
    if (!parser->parseFromFile(model_file.c_str(), verbosity)) {
        std::string msg("failed to parse onnx file");
        g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }

    // Build the engine
    builder->setMaxBatchSize(kBatchSize);
    // builder->setHalf2Mode(true);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 25);
    nvinfer1::ICudaEngine* engine =
        builder->buildEngineWithConfig(*network, *config);

    *engine_ptr = engine;
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
}


void PointPillars::EngineToTRTModel(
    const std::string &engine_file ,     
    nvinfer1::ICudaEngine** engine_ptr)  {
    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
    std::stringstream gieModelStream; 
    gieModelStream.seekg(0, gieModelStream.beg); // seekg 文件流对象读取，定位到begin位置，偏移值为0 

    std::ifstream cache(engine_file); 
    gieModelStream << cache.rdbuf();
    cache.close(); 
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger_); 

    if (runtime == nullptr) {
        std::string msg("failed to build runtime parser");
        g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }
    gieModelStream.seekg(0, std::ios::end);
    const int modelSize = gieModelStream.tellg(); // 定位当前位置

    gieModelStream.seekg(0, std::ios::beg);
    void* modelMem = malloc(modelSize); 
    gieModelStream.read((char*)modelMem, modelSize);


    std::cout << " |￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣> "<< std::endl;
    std::cout << " | " << engine_file << " >" <<  std::endl;
    std::cout << " |＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿> "<< std::endl;
    std::cout << "             (\\__/) ||                 "<< std::endl;
    std::cout << "             (•ㅅ•) ||                 "<< std::endl;
    std::cout << "             / 　 づ                    "<< std::endl;
    
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelMem, modelSize, NULL); 
    if (engine == nullptr) {
        std::string msg("failed to build engine parser");
        g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }
    *engine_ptr = engine;

}
// 推理主流程
void PointPillars::DoInference(const float* in_points_array,
                                const int in_num_points,
                                std::vector<float>* out_detections,
                                std::vector<int>* out_labels,
                                std::vector<float>* out_scores) 
/*
输入点云（CPU） ──(拷贝)──> dev_points（GPU）
       │
       └──(预处理: 体素化)──> dev_x_coors_/dev_pillar_point_feature_ 等（GPU）
       │
       └──(PFE 推理)──────> pfe_buffers_[1]（64维特征，GPU）
       │
       └──(特征散射)──────> dev_scattered_feature_（二维特征图，GPU）
       │
       └──(主干网络推理)──> rpn_buffers_[1]/[7]（检测原始结果，GPU）
       │
       └──(后处理: NMS)───> out_detections/out_labels/out_scores（CPU）
*/
{
    // if(in_points_array == nullptr) return;
    SetDeviceMemoryToZero();
    cudaDeviceSynchronize();
    // [STEP 1] : 加载点云到GPU
    float* dev_points = nullptr; // GPU内存
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_points),
                        in_num_points * kNumPointFeature * sizeof(float))); // [in_num_points , 5]
    GPU_CHECK(cudaMemset(dev_points, 0, in_num_points * kNumPointFeature * sizeof(float)));
    GPU_CHECK(cudaMemcpy(dev_points, in_points_array,
                        in_num_points * kNumPointFeature * sizeof(float),
                        cudaMemcpyHostToDevice));
    if (in_points_array == nullptr) {
        GPU_CHECK(cudaFree(dev_points)); // 提前释放
        return;
    }
    
    // [STEP 2] : preprocess
    host_pillar_count_[0] = 0; // 初始化有效体素计数为0（CPU端）

    // 记录预处理开始时间（用于性能统计）
    auto preprocess_start = std::chrono::high_resolution_clock::now();

    // 调用CUDA预处理模块，执行体素化
    preprocess_points_cuda_ptr_->DoPreprocessPointsCuda(
        dev_points,              // 输入：GPU上的点云数据（x,y,z,i,0）
        in_num_points,           // 输入：点云总数
        dev_x_coors_,            // 输出：体素X坐标（网格索引）
        dev_y_coors_,            // 输出：体素Y坐标（网格索引）
        dev_num_points_per_pillar_, // 输出：每个体素内的点数
        dev_pillar_point_feature_, // 输出：体素内的点特征矩阵
        dev_pillar_coors_,       // 输出：体素坐标范围（x_min, y_min, x_max, y_max）
        dev_sparse_pillar_map_,  // 输出：稀疏体素映射表（网格索引→体素索引）
        host_pillar_count_,      // 输出：有效体素总数（CPU→GPU同步）
        dev_pfe_gather_feature_  // 输出：收集后的体素点特征（供PFE输入）
    );
    // 等待CUDA核函数执行完成（同步GPU与CPU）
    cudaDeviceSynchronize();

    // 释放临时分配的GPU点云内存（预处理完成后不再需要原始点云）
    GPU_CHECK(cudaFree(dev_points));

    // 记录预处理结束时间
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    // DEVICE_SAVE<float>(dev_pfe_gather_feature_,  kMaxNumPillars * kMaxNumPointsPerPillar * kNumGatherPointFeature  , "0_Model_pfe_input_gather_feature");

    // [STEP 3] : pfe forward
    cudaStream_t stream;
    GPU_CHECK(cudaStreamCreate(&stream));
    auto pfe_start = std::chrono::high_resolution_clock::now();
    GPU_CHECK(cudaMemcpyAsync(pfe_buffers_[0], dev_pfe_gather_feature_,
                            kMaxNumPillars * kMaxNumPointsPerPillar * kNumGatherPointFeature * sizeof(float), ///kNumGatherPointFeature
                            cudaMemcpyDeviceToDevice, stream));
    pfe_context_->enqueueV2(pfe_buffers_, stream, nullptr);
    cudaDeviceSynchronize();
    auto pfe_end = std::chrono::high_resolution_clock::now();
    // DEVICE_SAVE<float>(reinterpret_cast<float*>(pfe_buffers_[1]),  kMaxNumPillars * 64 , "1_Model_pfe_output_buffers_[1]");

    // [STEP 4] : scatter pillar feature
    auto scatter_start = std::chrono::high_resolution_clock::now();
    scatter_cuda_ptr_->DoScatterCuda(
        host_pillar_count_[0], dev_x_coors_, dev_y_coors_,
        reinterpret_cast<float*>(pfe_buffers_[1]), dev_scattered_feature_);
    cudaDeviceSynchronize();
    auto scatter_end = std::chrono::high_resolution_clock::now();   
    // DEVICE_SAVE<float>(dev_scattered_feature_ ,  kRpnInputSize,"2_Model_backbone_input_dev_scattered_feature");

    // [STEP 5] : backbone forward
    auto backbone_start = std::chrono::high_resolution_clock::now();
    GPU_CHECK(cudaMemcpyAsync(rpn_buffers_[0], dev_scattered_feature_,
                            kBatchSize * kRpnInputSize * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream));
    backbone_context_->enqueueV2(rpn_buffers_, stream, nullptr);
    cudaDeviceSynchronize();
    auto backbone_end = std::chrono::high_resolution_clock::now();

    // [STEP 6]: postprocess (multihead)
    auto postprocess_start = std::chrono::high_resolution_clock::now();
    postprocess_cuda_ptr_->DoPostprocessCuda(
        reinterpret_cast<float*>(rpn_buffers_[1]), // [cls]   kNumAnchorPerCls 
        reinterpret_cast<float*>(rpn_buffers_[2]), // [cls]   kNumAnchorPerCls * 2 * 2
        reinterpret_cast<float*>(rpn_buffers_[3]), // [cls]   kNumAnchorPerCls * 2 * 2
        reinterpret_cast<float*>(rpn_buffers_[4]), // [cls]   kNumAnchorPerCls 
        reinterpret_cast<float*>(rpn_buffers_[5]), // [cls]   kNumAnchorPerCls * 2 * 2
        reinterpret_cast<float*>(rpn_buffers_[6]), // [cls]   kNumAnchorPerCls * 2 * 2
        reinterpret_cast<float*>(rpn_buffers_[7]), // [boxes] kNumAnchorPerCls * kNumClass * kNumOutputBoxFeature
        host_box_, 
        host_score_, 
        host_filtered_count_,
        *out_detections, *out_labels , *out_scores);
    cudaDeviceSynchronize();
    auto postprocess_end = std::chrono::high_resolution_clock::now();

    // release the stream and the buffers
    std::chrono::duration<double> preprocess_cost = preprocess_end - preprocess_start;
    std::chrono::duration<double> pfe_cost = pfe_end - pfe_start;
    std::chrono::duration<double> scatter_cost = scatter_end - scatter_start;
    std::chrono::duration<double> backbone_cost = backbone_end - backbone_start;
    std::chrono::duration<double> postprocess_cost = postprocess_end - postprocess_start;

    std::chrono::duration<double> pointpillars_cost = postprocess_end - preprocess_start;
    std::cout << "------------------------------------" << std::endl;
    std::cout << setiosflags(ios::left)  << setw(14) << "Module" << setw(12)  << "Time"  << resetiosflags(ios::left) << std::endl;
    std::cout << "------------------------------------" << std::endl;
    std::string Modules[] = {"Preprocess" , "Pfe" , "Scatter" , "Backbone" , "Postprocess" , "Summary"};
    double Times[] = {preprocess_cost.count() , pfe_cost.count() , scatter_cost.count() , backbone_cost.count() , postprocess_cost.count() , pointpillars_cost.count()}; 

    for (int i =0 ; i < 6 ; ++i) {
        std::cout << setiosflags(ios::left) << setw(14) << Modules[i]  << setw(8)  << Times[i] * 1000 << " ms" << resetiosflags(ios::left) << std::endl;
    }
    std::cout << "------------------------------------" << std::endl;
    cudaStreamDestroy(stream);

}
