# Task

在原始 FairGrad 代码基础上，实现一个最小侵入式的多任务训练框架，支持三层可组合模块：

1. Gradient preprocessing
   - identity
   - vargrad

2. Baseline solver
   - uniform
   - fairgrad
   - mgda
   - cagrad
   - nashmtl

3. Weight scheduler
   - every_step
   - psmgd_periodic

必须优先复用原项目的：

- 数据流
- trainer 入口
- 模型结构
- 数据集定义
- 日志
- delta_m / 评估逻辑
- stats 保存逻辑

不要推倒重写。

---

# Required Pipeline

每个 step 必须严格按下面顺序执行：

1. 计算每个任务的原始梯度 `g_t^k`
2. 做梯度预处理
3. 用 baseline solver 计算候选权重 `lambda_hat`
4. 用 scheduler 生成当前权重 `lambda_t`
5. 聚合梯度并更新参数

---

# Math

## Raw gradient

对每个任务 k：

$$
g_t^k = \nabla_{\theta} L_t^k
$$

必须显式拿到单任务梯度，不能用总 loss 的混合梯度代替。

## VarGrad

如果启用：

$$
c_t^k = g_t^k + \frac{\beta_v}{1 - \beta_v}\left(g_t^k - g_{t-1}^k\right)
$$

否则：

$$
c_t^k = g_t^k
$$

## Momentum

如果启用：

$$
m_t^k = \beta_m m_{t-1}^k + (1 - \beta_m) c_t^k
$$

否则：

$$
m_t^k = c_t^k
$$

## Solver

solver 输入默认使用 `m_t^k`。

solver 输出候选权重：

$$
\hat{\lambda}_t
$$

要求：

- `fairgrad` 必须沿用原始 FairGrad 项目的定义
- 不要重新定义 FairGrad 数学公式
- `uniform` 使用均匀权重
- `mgda/cagrad/nashmtl` 尽量复用原项目实现

## Scheduler

### every_step
如果不启用 PSMGD：

$$
\lambda_t = \hat{\lambda}_t
$$

### psmgd_periodic
如果启用 PSMGD：

- 当 `t % R == 0`：
  $$
  \lambda_t = \alpha \lambda_{\mathrm{prev\_cycle}} + (1 - \alpha)\hat{\lambda}_t
  $$
- 当 `t % R != 0`：
  $$
  \lambda_t = \lambda_{t-1}
  $$

最后始终归一化 `lambda_t`。

## Update

$$
g_t^{\mathrm{agg}} = \sum_k \lambda_t^k m_t^k
$$

用 `g_t^agg` 更新共享参数。

---

# Strict Separation

必须保持三层职责解耦：

## preprocessing
只负责：
- `g -> c`
- `c -> m`

## solver
只负责：
- 根据当前任务梯度生成 `lambda_hat`

## scheduler
只负责：
- 根据 `lambda_hat`、历史权重、step 生成 `lambda_t`

不要把：

- VarGrad 写进 solver
- PSMGD 写进 solver
- solver 写进 VarGrad

---

# Compatibility Rules

## Original FairGrad compatibility

当：

- preprocessing = identity
- solver = fairgrad
- scheduler = every_step

时，行为应尽量接近原始 FairGrad。

## Baseline fallback

当：

- preprocessing = identity
- solver = uniform
- scheduler = every_step

时，应退化为默认均匀权重 baseline。

---

# Implementation Preference

优先在这些位置扩展：

- `methods/weight_methods.py`
- 参数解析
- 原始 trainer 调用链
- 原始 run scripts

避免新建一整套并行 trainer。

---

# Required Experiment Modes

至少支持以下组合：

## solver only
- fairgrad
- mgda
- cagrad
- nashmtl
- uniform

## vargrad + solver
- vargrad + fairgrad
- vargrad + mgda
- vargrad + cagrad
- vargrad + nashmtl

## solver + psmgd
- fairgrad + psmgd
- mgda + psmgd
- cagrad + psmgd
- nashmtl + psmgd

## full
- vargrad + fairgrad + psmgd
- vargrad + mgda + psmgd
- vargrad + cagrad + psmgd
- vargrad + nashmtl + psmgd

---

# Config Requirements

至少支持这些配置项：

- `preprocessing`
- `solver`
- `scheduler`
- `use_vargrad`
- `use_momentum`
- `use_psmgd`

以及这些超参数：

- `beta_v`
- `beta_m`
- `psmgd_R`
- `psmgd_alpha`

---

# Non-Goals

不要实现以下内容，除非明确要求：

- VarGrad 里的 SMO
- 新定义的 FairGrad 数学公式
- 重写新的训练系统
- 重写数据集和模型
