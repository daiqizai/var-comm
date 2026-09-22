# 联合梯度边界与初始化末位差

准备阶段第一次CPU检查中，权重SHA相同，但将E从`requires_grad=False`改为True后，batch4的FP32发送波形没有逐bit相同；最大差2.98e-7。该失败发生在新合成检查，没有开始新GPU训练，旧数据/配置/SHA均未更改。

已定位至非连续3D输入的`channel_lift`：该层输出最大差7.15e-7；仅在独立诊断中将输入设为连续时差为0，FP64完整波形也相同。生产模型没有为了过检查添加`contiguous`或修改旧代码。

本机torch为2.11.0+cu128，git版本`70d99e998b4955e0049d13a98d77ae1b14db1f45`。核对该版本官方`aten/src/ATen/native/LinearAlgebra.cpp`的`should_fold`/`_matmul_impl`：小矩阵的梯度标志可以改变folded mm与bmm分派。原始来源`https://raw.githubusercontent.com/pytorch/pytorch/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/LinearAlgebra.cpp`。

因此不把“权重相同”错误扩大成“梯度标记不同也必然FP32逐位相同”。严格保持权重SHA检查及冻结模式的逐位回放；Joint波形采用此前已存在于RX评测配置的1e-6回放容差，并独立核对FP64数学一致性。未放宽任何冻结实验的阈值或校验值；真实GPU检查仍待执行。

训练中的接收梯度边界另作处理：R四个micro均读取同一次batch4生成的波形值，在叶子波形累加梯度；最后对原`received`图执行一次带该合计梯度的backward。链式法则覆盖图像→R→信道→E，避免四次重复反传同一个batch4发送图。此为精确微批次梯度分解，不是将离散选择当可微的ST近似。
