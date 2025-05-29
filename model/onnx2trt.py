import tensorrt as trt
import os

# 定义日志记录器
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, fp16_mode=True, workspace_size=1024):
    """
    将 ONNX 模型转换为 TensorRT 引擎
    
    参数:
    onnx_path (str): ONNX 模型路径
    engine_path (str): 保存 TensorRT 引擎的路径
    fp16_mode (bool): 是否使用 FP16 精度
    workspace_size (int): 工作空间大小(MB)
    """
    # 创建构建器
    builder = trt.Builder(TRT_LOGGER)
    
    # 配置网络
    network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_creation_flag)
    
    # 创建 ONNX 解析器
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 读取 ONNX 文件
    with open(onnx_path, 'rb') as model:
        print('正在解析 ONNX 模型...')
        if not parser.parse(model.read()):
            print('解析 ONNX 模型失败')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
        print('ONNX 模型解析成功')
    
    # 构建器配置
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size * 1024 * 1024  # 转换为字节
    
    # 启用 FP16 模式
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print('已启用 FP16 精度')
    
    # 移除不支持的 BEST_PRECISION 标志
    # config.set_flag(trt.BuilderFlag.BEST_PRECISION)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    # config.set_flag(trt.BuilderFlag.PREFER_PRECISION_REDUCTION)
    print("启用最佳精度与性能优化策略")
    # print('已移除不支持的 BEST_PRECISION 配置')
    
    # 构建引擎
    print('正在构建 TensorRT 引擎... 这可能需要几分钟时间')
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print('构建 TensorRT 引擎失败')
        return None
    print('TensorRT 引擎构建成功')
    
    # 保存引擎
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
        print(f'TensorRT 引擎已保存至 {engine_path}')
    
    return