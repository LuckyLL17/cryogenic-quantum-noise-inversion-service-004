# 低温量子器件实验数据校准与噪声谱反演服务

纯后端 Node.js + TypeScript 服务，不依赖数据库、Docker 或第三方运行时。服务从多通道 ADC/IQ 采样、实验时钟、仪器配置和校准版本中重建实验窗口，完成时钟同步、IQ 校准、窗口化、功率谱/互谱计算、相干性分析、噪声类型判别和 T1/T2/Ramsey 等衰减参数拟合。

## 运行

```bash
npm test
npm start
```

默认监听 `127.0.0.1:4387`。

## 主要接口

- `GET /health`
- `POST /v1/analyses`：提交分析任务，返回任务 ID
- `GET /v1/analyses/:id`：读取任务状态或结果
- `POST /v1/analyses/:id/cancel`：取消排队或运行中的任务
- `POST /v1/experiments/preview`：同步、校准并快速预览派生指标

服务只使用内存任务存储；任务结果通过版本化的输入快照和校准版本引用保证可追溯。

## 衰减拟合结果（metrics.decay）

T1/T2/Ramsey 的 `decay` 不再保证总有 `timeConstantNs`，而是先给出 `status`：

- `fitted`：识别到指数衰减，`timeConstantNs`、`amplitude0`、`confidenceInterval` 有效
- `baseline-drift`：基线在拟合窗口内显著漂移；衰减仍可辨识时仍给出 `timeConstantNs`（`confidence: low`，原因 `baseline-drift-corrected`），否则只给 `baselineSlopePerNs`
- `non-decaying`：曲线在噪声内无可见衰减（平直、仅趋势或上升），`timeConstantNs` 为 null
- `insufficient-points`：可用内点不足 3 个（`usedPoints`/`rejectedPoints` 仍给出）
- `invalid`：窗口内无样本或数值无法拟合，绝不返回伪造的时间常数

离群点（准备阶段残留、毛刺）由 LMedS 种子 + Huber/Tukey 加权迭代按数据自身残差尺度剔除，不使用固定幅值阈值；常数、线性漂移、恒基线衰减、漂移基线衰减四个模型通过 BIC、似然比检验和信噪比门限选择。`reasons` 给出降级依据（如 `wide-confidence-interval`、`time-constant-unconstrained`、`structured-residuals`）。`fitStartNs`/`fitEndNs`、`usedPoints`/`rejectedPoints` 与既有 `noise-only`（不产出 decay）行为保持不变。
