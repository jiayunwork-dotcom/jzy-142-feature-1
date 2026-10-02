# 弹性反应谱计算与谱匹配作业服务

把一批强震加速度记录提交上来，后台用 Newmark-β 逐步积分计算每条记录的
弹性反应谱（SD / SV / SA / PSV / PSA），再按调用方给出的设计谱做
周期区间内的对数误差最优缩放与匹配筛选，输出排序、前 N 条平均谱及
逐点比值。上传、计算、匹配都是异步作业，可查进度 / 部分结果 / 取消；
记录、作业状态与结果全部落 SQLite，进程重启后已完成结果照常可读，
执行中的作业标记为中断。

**新版规范（双向时程成对选波）能力**：把同一台站同一次地震的两条
水平分量作为「分量组」（component group）一等对象管理——显式建组或
按文件头信息获取配对建议（建议只提示、不自动建组）；自动统一单位、
对齐时间网格；后台作业计算与方向无关的组合谱 **RotD50 / RotD100**
（位移 SD、绝对加速度 SA、伪加速度 PSA 三类量，附 RotD100 方位角、
两条原始分量谱及其几何平均）；匹配作业以组为单位，两条分量**共用一个
缩放系数**、按组合谱对设计谱，输出每组缩放后两条分量各自的谱。

- Python 3.12 + Flask + NumPy
- 积分器、反应谱、组合谱、匹配优化均为自实现（仅线性代数借助 NumPy）
- 容器：`python:3.12-slim`，gunicorn 启动，SQLite 在挂载卷

---

## 1. 目录结构

```
app/
  errors.py       # 领域异常（逐条可读错误）
  parsing.py      # 记录文本解析（两列 / 单列+步长、文件头元数据）
  baseline.py     # 基线校正（none/mean/linear）与速度位移积分
  integrator.py   # Newmark-β 积分器（向量化、自动加密、自由振动段）
  spectrum.py     # 单条弹性反应谱 SD/SV/SA/PSV/PSA、零周期点、PGA/PGV/PGD
  rotd.py         # 分量组组合谱 RotD50/RotD100（流式积分 + 角度收缩，独立模块）
  groups.py       # 分量组：单位统一、时间网格对齐（有理网格/插值/截齐/补零）
  pairing.py      # 按文件头 station/event/component 给配对建议（只建议）
  matching.py     # 设计谱对数插值、最优缩放、越限剔除、排序、平均谱比值
                  #   （match_batch 单条 / match_group_batch 分量组）
  jobs.py         # spectrum / match 两类单条作业的执行逻辑
  group_jobs.py   # rot_spectrum / match_group 两类分量组作业 + 结果缓存
  scheduler.py    # 单工作线程后台调度、崩溃恢复
  storage.py      # SQLite：records / jobs / job_items / component_groups / rot_spectra
  api/
    records.py    # 记录上传与查询
    jobs.py       # 单条作业创建 / 查询 / 取消
    groups.py     # 分量组、配对建议、组合谱/组匹配作业
    helpers.py
  __init__.py     # Flask 应用工厂
wsgi.py           # gunicorn 入口
tests/            # pytest（含旧版本库文件升级启动测试 tests/test_upgrade.py）
```

## 2. 快速开始

### 本地

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export SEISMIC_DB=/tmp/seismic.db
gunicorn -c gunicorn.conf.py wsgi:app
# 或开发模式：flask --app wsgi run
```

### 容器

```bash
docker compose up --build -d
curl -s http://localhost:8000/health
```

SQLite 文件位于卷 `seismic-data` 容器内 `/data/seismic.db`。

### 运行测试

```bash
pip install -r requirements.txt
pytest -q
```

---

## 3. 上传记录

### multipart/form-data

字段名 `files` 可重复，每个文件一条；表单字段 `unit`（`m/s2|g`）整批
生效，`dt` 作为单列格式默认步长。

```bash
curl -s -X POST http://localhost:8000/api/records \
  -F 'files=@RSN1.txt' -F 'files=@RSN2.txt' \
  -F 'unit=m/s2'
```

### application/json

```json
{
  "unit": "g", "dt": 0.01,
  "records": [
    {"name": "RSN1.AT2", "content": "# Station: xxx\n0.00 0.001\n..."},
    {"name": "single.txt", "content_b64": "...", "dt": 0.005}
  ]
}
```

单条可用 `unit` / `dt` 覆盖整批默认值。

### 支持的文件格式

1. **时间–加速度两列**：`t a`，按数据核对等步长（相对容差 1e-4），
   声明 `dt` 时还要与时间列一致；
2. **序号–加速度两列**：第一列为 0,1,2,… 整数序号，此时必须显式给 `dt`；
3. **单列加速度**：必须给 `dt`（请求参数，或文件头 `# DT: 0.005 SEC`）。

空白/逗号/分号分隔均可。`# % * ! /` 开头的行视为文件头，原文保留到
`metadata.header`，并抽取 `station / component / channel / event /
dt / unit / name` 等常见关键字。

### 单位（必须声明）

加速度单位只接受 **`m/s2` 或 `g`**（按 9.80665 换算）。未声明单位的
记录单条标记为 `error`，不影响同批其他记录。`gal / cm/s2` 直接拒绝
（量级容易搞错），请先换算再上传。

### 逐条错误（绝不让整批一起失败）

空文件、非数值行、列数不一致、步长不一致、单位缺失或无法识别、
NaN/Inf、超过 20 万点等，都在对应记录上给出带行号的可读错误，失败
记录同样入库（`status=error`），便于事后排查。响应形如：

```json
{"uploaded": 3, "ready": 1, "failed": 2,
 "records": [
   {"name": "bad.txt", "status": "error",
    "error": "第 2 行含有非数值数据 'oops'，请检查文件内容或分隔符",
    "record_id": "rec_bad..."},
   ...
 ]}
```

限制：一批最多 **200** 条，单条最多 **200 000** 点（超出整请求 400
或该条 error）。

记录 ID 由文件名 + 加速度字节 + 步长/单位的 SHA-256 前 16 位决定，
重复上传同一文件得到同一 ID。

---

## 4. 创建作业

### 4.1 反应谱作业

`POST /api/jobs/spectrum`

```json
{
  "record_ids": ["rec_...", "rec_..."],
  "periods": [0.0, 0.02, 0.05, 0.1, 0.5, 1.0, 3.0, 6.0],
  "dampings": [0.02, 0.05],
  "method": "average_acceleration",
  "instability_policy": "refine",
  "baseline_mode": "mean"
}
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `periods` | `logspace(log10 0.02, log10 6, 100)` | 严格递增、非负；允许包含 0 |
| `dampings` | `0.05` | 数值或列表，均须 ∈ [0,1] |
| `method` | `average_acceleration` | `average_acceleration`（γ=1/2,β=1/4，无条件稳定）或 `linear_acceleration`（γ=1/2,β=1/6） |
| `instability_policy` | `refine` | `refine`（线性插值自动加密）/ `reject`（失稳即该条报错） |
| `baseline_mode` | `mean` | `none` / `mean` / `linear`，见 §7 |

### 4.2 匹配作业

`POST /api/jobs/match`，在谱作业参数之外增加：

```json
{
  "record_ids": ["rec_a", "rec_b"],
  "periods": [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0],
  "dampings": [0.05],
  "target_damping": 0.05,
  "design_spectrum": {
    "periods": [0.01, 0.1, 0.5, 1.0, 6.0],
    "values":  [0.20, 0.60, 0.60, 0.35, 0.10],
    "unit": "g"
  },
  "t1": 0.1, "t2": 2.0,
  "s_min": 0.5, "s_max": 3.0,
  "top_n": 7,
  "quantity": "psa",
  "bounds_policy": "reject"
}
```

- 设计谱控制点周期严格递增、周期与谱值严格为正（对数插值要求）；
- 匹配区间 `[t1,t2]` 必须落在设计谱定义范围内；
- `quantity`：`psa`（默认，伪加速度）或 `sa`；
- 设计谱单位为 `g` 时记录谱自动换算到 g 后再匹配；
- `bounds_policy`：`reject`（越限即从候选剔除并注明原因）或 `clamp`
  （取边界系数重算误差后仍参与排序）。

### 查询与取消

| 接口 | 说明 |
|---|---|
| `GET /api/jobs` | 最近作业（不含大结果体） |
| `GET /api/jobs/<id>` | 状态、进度、完成后的完整结果 |
| `GET /api/jobs/<id>/items` | 逐条作业项，执行中即可看到已完成部分 |
| `POST /api/jobs/<id>/cancel` | 取消排队 / 执行中的作业 |

作业状态：`queued → running → completed / failed / cancelled`；
进程重启时上次 `running` 的作业变为 `interrupted`（结果不可用，
请重新提交），`queued` 作业继续排队执行。

---

## 5. 输出内容

每条谱结果包含（每个阻尼比一组）：

- `periods`；
- `sd`（相对位移峰值，m）、`sv`（相对速度峰值，m/s）、
  `sa`（绝对加速度峰值，m/s²）；
- `psv = ω·SD`、`psa = ω²·SD`；
- `sa_g`、`psa_g`：以 g 计，便于直接对规范谱；
- `method`、`dt_used`、`refined`、`refine_factor`、
  `unstable_at_input_dt`（逐周期布尔掩码）、`tail_steps`——**明确写出
  线性加速度法是否触发了加密、加密倍数与实际积分步长**；
- 记录层面：`pga / pgv / pgd`（及 `pga_g`）、采用的 `baseline` 方式、
  自由振动段信息 `free_vibration.included=true, periods_factor=1.25`。

匹配结果：`ranking`（rank、record_id、scale、无约束最优
`scale_optimal`、`mse`、`rmse_log`、是否越限）、`excluded`（越限记录
及原因）、`average_spectrum`（前 N 条缩放后谱的算术平均与几何平均）、
`ratios`（设计谱控制点上的平均谱/几何平均谱与设计谱的**逐点比值**）。

---

## 6. 数值方法

### 6.1 运动方程与 Newmark-β

单位质量单自由度线弹性体系：

```
ẍ + 2ξω ẋ + ω²x = -a_g(t)
```

静止起步 `x(0)=v(0)=0`，故 `a(0) = -a_g(0)`。每个时间步解等效
静力方程 `K̂ x_{n+1} = RHS`，再由 Newmark 假设回代 `a_{n+1}`、
`v_{n+1}`；所有振荡器（不同周期/同一阻尼）作为数组最后一维做向量化
递推。绝对加速度取 `a_rel + a_g`。

### 6.2 稳定性与自动加密

- 平均加速度法（β=1/4）无条件稳定；
- 线性加速度法（β=1/6）条件稳定，无阻尼临界
  `ωΔt ≤ √12`，即 **`Δt/T ≤ √12/(2π) ≈ 0.55133`**（这是最严的
  无阻尼判据，对 ξ≥0 同样安全）；
- 越限时 `refine` 策略按 `ceil((Δt/Tmin)/0.55133)` 把输入**线性插值**
  加密为整数倍子步——线性加速度法本身假设区间内加速度线性变化，
  因此该加密与其离散假设严格一致，不引入额外近似；`reject` 策略则
  直接让该条记录失败并在错误中写明周期、Δt/T 与上限。

### 6.3 自由振动段

记录结束并不意味着响应峰值已经出现。统一在强迫段后补算
**1.25 个（本批最长）自振周期**的零输入自由振动（至少 1 步）并
纳入 SD/SV/SA 峰值统计。

为什么 1.25 个周期对 0≤ξ≤1 都足够：自由振动可写成
`x = e^{-ξωt}(A sin ωd t + B cos ωd t)`，相邻局部极值幅值以
`e^{-ξωΔt}` 衰减，而首个极值落在 `t∈[0, Td/2]` 内。自由振动
**初始包络**为

```
R0 = sqrt(x0² + ((v0+ξωx0)/ωd)²),  ωd = ω sqrt(1-ξ²)
```

它就是自由段 `|x|` 的理论最大值（ξ=0 时恰为一个自振周期内扫到的
振幅），故补满略大于一个周期（取 1.25 留边界余量）即可保证不遗漏
峰值；有阻尼时峰值出现更早、后续指数衰减，更长尾部不会改变峰值。
这也保证「在记录末尾再补一段零，谱值不变」（见测试）。

### 6.4 零周期点与极限行为

- 周期列表允许含 0：该点 `SA = PSA = PGA`，`SD = SV = PSV = 0`；
- T→0 时 SA→PGA（体系刚度无穷大，质量与地面同加速度）；
- T→∞ 时 SD→PGD（体系近不动，相对位移即地面位移），要求记录先做
  基线处理（见 §7）。

### 6.5 谱匹配优化

统计网格取记录谱周期点 ∩ [t1,t2]，并补入区间端点；设计谱在该网格
按**对数–对数插值**。对缩放系数 s 最小化

```
Σ_j [ ln(s A_j) - ln D_j ]²
```

令导数为零得闭式最优解（各点比值的几何平均）：

```
s* = exp( mean_j ln(D_j / A_j) )
mse = mean_j ( ln(s A_j / D_j) )²
```

这是一维无约束凸问题的精确解（无需迭代优化器）；`clamp` 时取
`s = clip(s*, s_min, s_max)` 后重算 mse。记录整体乘 k 时，
`A→kA ⇒ s*→s*/k`，mse 不变（测试逐条验证）。

### 6.6 确定性

相同输入 → 相同结果：记录 ID 内容寻址；匹配排序在 mse 相同时按
record_id 二次排序；结果 JSON 键排序序列化；不依赖当前时间作为
随机源。重复提交同一批作业，结果除作业 ID/时间戳外完全一致。

---

## 7. 基线处理（重点说明）

强震加速度记录常带微小零点偏移/慢漂移。两次梯形积分求地面位移时，
恒定偏移 ε 会产生 `½ ε t²` 量级的**虚假位移漂移**，直接破坏
「长周期 SD 趋于 PGD」。因此每条记录计算前统一做基线校正，方式写入
请求参数 `baseline_mode` 与结果 `baseline` 字段：

| 方式 | 操作 | 用途 |
|---|---|---|
| `none` | 不处理 | 记录已离线校正过时使用 |
| `mean`（默认） | 减去全程加速度均值（去直流） | 一般记录；处理后末端地面速度回到 ≈0 |
| `linear` | 最小二乘拟合一次趋势线并扣除（同时去直流 + 去线性漂移） | 末端速度/位移明显不回零的记录 |

地面速度、位移均由复合梯形积分、零初始条件得到：

```
v[0]=d[0]=0,  v[i]=v[i-1]+(a[i-1]+a[i])·Δt/2  （再积分一次得 d）
PGV = max|v|,  PGD = max|d|
```

基线校正属于线性运算：在默认 `mean` 下记录整体乘 k 时谱仍线性
（仅有求和顺序的浮点舍入，测试以 1e-10 容差验证；`none` 模式
逐位线性）。

---

## 8. 作业调度与持久化

- 单工作线程**串行**执行作业（SQLite 单写；NumPy/BLAS 负责计算并行），
  每条记录处理前轮询取消标志；
- 单条记录失败只标记该作业项 `error/skipped`（带可读原因），作业
  整体照常 `completed`，`result.summary` 给出 done/error/skipped 计数；
- SQLite（WAL）三张表：`records`（含加速度 BLOB、元数据 JSON、
  status/error）、`jobs`（参数/进度/结果 JSON/时间戳）、`job_items`
  （逐条状态与结果，执行中即可查部分结果）；
- 进程重启：`running → interrupted`（注明原因），`queued` 继续执行；
  `completed` 结果长期可查。

gunicorn 刻意配置为 **1 worker + 多线程**，避免多进程各自启动调度
线程重复领取队列（见 `gunicorn.conf.py` 注释）。

---

## 9. 测试覆盖（pytest，共 140+ 用例）

需求中列出的关系均有**独立测试**逐条确认：

- 单条：记录整体乘 k 谱乘 k、匹配系数除以 k
  （`test_scaling_linearity*`、`test_scaled_record_scales_inverse`）；
- **分量组组合谱（`tests/test_rotd.py`）**：
  - 两条分量整体绕竖轴旋转 30° / 117° 后，RotD50/RotD100 不变，
    RotD100 方位角整体平移（`test_rotation_invariance*`）；
  - 交换两条分量、把其中一条整体取反，组合谱不变；
  - 一条分量全零，RotD100 恰等于另一条单谱（RotD50 解析上为 1/√2）；
  - 两条分量完全相同，RotD100 = √2 × 单条谱；
  - 逐周期 RotD100 ≥ RotD50、RotD100 ≥ 两条分量谱较大者（SD/SA/PSA）；
  - 整组乘 k 组合谱乘 k、组匹配缩放系数变为 1/k；
  - 同组同参数重复提交**逐位相同**；
  - 180 角度网格相对 2880 网格的离散误差不超过文档声明的 0.87%/1%；
- **分量组与对齐（`tests/test_groups.py`）**：单位统一、有理网格不插值、
  无理比线性插值、过粗加密拒绝、时间窗不重叠拒绝、union 补零/
  intersection 截齐及补零计数、大时差告警、配对建议只建议不建组、
  跨台站不混配；
- **端到端（`tests/test_groups_api.py`）**：建议→确认建组、组合谱作业、
  缓存命中逐位一致、单组坏不拖垮整批、组匹配共用缩放系数与成对缩放谱、
  整组乘 k 系数 1/k、取消、批量上限、重启后组仍在；
- **升级（`tests/test_upgrade.py`）**：用旧版本结构手工建库，新版直接
  打开后老记录/老作业/老结果字段不变、running→interrupted、queued 续跑、
  新能力可用；
- 纯正弦地面运动稳态放大系数与解析解
  `R(r)=1/sqrt((1-r²)²+(2ξr)²)` 在 r=0.7/1.0/1.3 相对误差 <1%
  （`test_sine_steady_state_amplification_within_1pct`）。测试信号
  用 20 周期半余弦包络平滑起步以消除**起振瞬态拍振**，并按自由振动
  初始包络 R0 的二次型最小特征值选择截断相位，保证记录突停后的自由
  振动不抬高峰值——这两个构造的物理原因在测试 docstring 中写明；
- ξ=0 时 SA 恰等于 PSA（容差 1e-10），小阻尼中短周期段两者统计接近；
- T→0 时 SA→PGA（带限信号严格验证；并注释了宽带噪声短周期 SA 高于
  PGA 是真实高频反力现象）；T→∞ 时 SD→PGD（位移脉冲构造）；
- 两种 Newmark 参数在步长足够细时收敛到同一结果（<2e-3）；
- 记录末尾补一段零，谱值不变；
- 同一批作业重复提交，结果完全一致；
- 线性加速度法稳定判据、自动加密倍数、`reject` 报错信息、
  `unstable_at_input_dt` 掩码；
- 解析：空文件、非数值行（带行号）、步长不一致、单位缺失/不支持、
  阻尼越界、周期非递增、设计谱非法、缩放越限剔除、批量上限；
- 存储往返、崩溃恢复（running→interrupted、queued 续跑）、
  完成结果跨「重启」可读、HTTP 端到端上传/建作业/进度/取消/部分结果。

## 10. 容量与性能

- 批量 ≤ 200 条、单条 ≤ 200 000 点（上传与建作业两处都校验）；
  这两个上限对**分量组**同样适用：一批最多 200 组、对齐后每组最多
  200 000 点；
- 典型强震记录（数千点、100 周期点、5% 阻尼、180 方位角）单组亚秒级；
- 20 万点 × 100 周期的极端组合每步是 100 维向量运算，主要成本为
  Python 时间步循环，属后台批处理可接受范围；积分热点全部在 NumPy。

---

## 11. 分量组（双向时程成对管理）

一组由**同一台站、同一次地震的两条水平分量**组成；可另挂一条竖向
分量（只保留引用，不参与水平组合）。

### 11.1 配对建议（只建议，不自动建）

```bash
curl -s -X POST http://localhost:8000/api/groups/suggestions \
  -H 'Content-Type: application/json' \
  -d '{"record_ids": ["rec_...", "rec_..."]}'   # record_ids 可省略=全库扫描
```

按上传时文件头抽取的 `station / event / component(channel)` 分桶配对：
跨台站/跨事件绝不配到一起；桶内按方位角正交（差 ≈90°，容差 15°）给
`preferred` 建议，角度读不出但桶内恰有两条水平分量给 `possible`，
无法确定的进 `unpaired` 并写明原因；桶内恰有一条竖向分量时自动作为
`vertical_id` 附上。**服务绝不据此自动建组**，必须调用方确认后显式
POST（响应里也带这条提示）。

### 11.2 显式建组

```bash
curl -s -X POST http://localhost:8000/api/groups \
  -H 'Content-Type: application/json' -d '{
    "h1_id": "rec_...", "h2_id": "rec_...",
    "vertical_id": "rec_...",          # 可省
    "name": "ST-A/EV1",                # 可省
    "trim": "union"                    # 默认 union，可选 intersection
}'
```

- 两条分量单位不同（m/s² 与 g）时统一换算到 **m/s²**（按 9.80665）；
- 组 ID 内容寻址：成员（**无序**，交换 h1/h2 得到同一组）+ 对齐后
  序列字节 + dt + 竖向引用的 SHA-256，重复建组幂等；
- 组与对齐后序列一起落 `component_groups` 表（BLOB），重启后仍在，
  重复计算逐位一致。查询：`GET /api/groups`、`GET /api/groups/<id>`
  （列表/详情不回传 BLOB）。

### 11.3 分量对不齐时怎么办（明确的取舍）

对齐规则全部写进组信息 `alignment`（每分量的输入步长/点数/起始时刻/
是否被重采样/首尾补零点数 + 告警），拒绝时返回带具体数值的可读原因。

1. **起始时刻**：从文件头 `start_time/t0/...` 读取（读不到按 0）。
   两段时间窗**完全不重叠**（重叠不足一个步长）直接拒绝；起始时差
   超过较短记录持时一半时**照常对齐但给出告警**（提示可能错配）。
2. **步长不一致**：以**较粗步长为基准**找小整数细分网格
   `p·dt1 ≈ q·dt2`（p、q ≤ 200，相对容差 1e-6）。此时每条分量只落在
   公共网格的整数点上（如 40Hz/50Hz → 5:4、dt₀=0.005s），**不产生
   插值误差**；找不到这种有理网格时，以较细步长为公共网格、对较粗
   分量做**线性插值**（线性加速度法本身也假设区间内加速度线性），
   加密倍数 >100 直接拒绝。
3. **时长/点数不同**：默认 `trim="union"`，取两时间窗并集、各自覆盖
   不到的网格点**补零**（该方向在那段时间无地面输入，自由振动段由
   积分器统一另补）；`trim="intersection"` 只保留重叠段、两端截齐。

   **为什么默认补零而不是一律截齐**：截齐会丢掉先到/后到段里真实的
   地震动能量（强震有效持时通常远大于台站触发时差的影响）；补零不产生
   强迫响应、保留全部真实输入。代价是错配两条触发时刻相差整段持时的
   记录时补零会掩盖错配——因此有上面的大时差告警与不重叠硬拒绝。
4. 对齐后点数 > 200 000 拒绝。

---

## 12. 组合谱 RotD50 / RotD100 与组匹配

### 12.1 创建组合谱作业

`POST /api/jobs/rot-spectrum`

```json
{
  "group_ids": ["grp_..."],
  "periods": [0.0, 0.02, 0.05, 0.1, 0.5, 1.0, 3.0, 6.0],
  "dampings": [0.02, 0.05],
  "n_angles": 180,
  "method": "average_acceleration",
  "instability_policy": "refine",
  "baseline_mode": "mean"
}
```

除 `n_angles`（默认 180，允许 8–3600）外参数与单条谱作业完全一致。
作业机制也一致：后台执行、`GET /api/jobs/<id>` 查进度与完整结果、
`/items` 查逐组成果、`/cancel` 取消；一批里某组算不出来只把该组标记
为 `error/skipped`，其余照常，`summary` 给 done/error/skipped/
cache_hits 计数。

每个阻尼比的结果含：

- `rotd50` / `rotd100`：各带 **`sd`（相对位移）、`sa`（绝对加速度）、
  `psa`（伪加速度）**及 g 单位字段 `sa_g/psa_g`；
- `rotd100_angle_deg`：逐周期取得 RotD100 的方位角（度，0–180）；
- `components`：**两条原始分量各自的谱**（sd/sa/psv/psa + PGA/PGV/PGD）；
- `geomean`：两条分量谱的逐点几何平均（对照口径）；
- `method/dt_used/refined/refine_factor/unstable_at_input_dt/tail_steps`
  与单条谱口径一致；T=0 点 SA=PSA=PGA（对地面加速度向量做角度收缩）。

### 12.2 方位角离散、计算组织与误差上限（三处取舍之一）

定义：对方位角 θ（水平投影轴），先求该方向 SDOF 体系的响应时程、取
峰值 `R(θ)`；`RotD100 = max_θ R(θ)`、`RotD50 = median_θ R(θ)`。
θ 与 θ+180° 只差正负号、峰值相同，故 θ 只在 **[0°,180°)** 取，默认
180 点（**1° 一格**）。

**计算组织**：线性系统满足叠加原理，θ 方向响应
`xθ(t) = x₁(t)cosθ + x₂(t)sinθ`，因此**每条分量只积分一次**，角度
投影在峰值收缩阶段完成，而不是逐角度重新积分（后者耗时随角度数成倍
增长）。分量 Newmark 递推按时间块流式产出，角度按每批 32 个投影、
在块内收缩成峰值累加器（形状仅「周期 × 角度」，100×180×8 B≈144 KB），
**任何时刻都不保存「时间 × 周期 × 角度」全量中间量**。代价是丢失
逐角度的完整时程（本服务只需要各角度峰值，不需要），以及角度收缩阶段
的 NumPy 运算量（512×100×32 的块运算，实测远小于积分循环）。

**误差上限**：组合峰值是集合 `K = conv{±r(t)}` 的支撑函数。角度网格
只取真方向 ±Δθ/2 内的方向，网格最大值对真最大值的相对低估不超过

```
1 − cos(Δθ/2)/(1 + sin(Δθ/2))
```

Δθ=1° 时 ≈ **0.87%**——这是 RotD100 的**硬上界（只会低估）**；RotD50
取中位、更平滑，服务对 RotD50 声明 **1%** 的保证容差。180 点网格对
典型强震记录相对 2880 点参考网格的实测偏差：RotD100 < 4×10⁻⁵、
RotD50 < 1×10⁻³（测试 `test_angle_discretization_within_declared_bound`
逐条用 2880 点参考核对，容差即上面的声明值，未为通过测试放宽）。
1° 网格还保证**整体旋转任意整数度数后角度集合只是置换**（旋转不变性
测试 30°、117°）。需要更小时可调大 `n_angles`（最大 3600，即 0.05°）。

**峰值内存**（本机实测 maxRSS，200 000 点上限）：
普通 dt 下平均加速度法最大驻留约 **65 MB**；最坏情形（dt=0.055s 触发
线性加速度法 5× 自动加密、100 周期×180 角、20 万点）maxRSS 约
**140 MB**（两份加密序列 + 输入约 25 MB，分块投影临时量 + Python/NumPy
基线约 110 MB；该极端组合耗时约 140 s），不会撑爆容器。另设 500 万
加密步的硬保护上限（超过直接报错而不是继续申请内存）。

### 12.3 组匹配作业

`POST /api/jobs/match-group`，参数在 12.1 基础上与单条 match 一致，
另加 `rotd_band`（`rotd50` 默认 / `rotd100`）：

```json
{
  "group_ids": ["grp_a", "grp_b"],
  "target_damping": 0.05,
  "design_spectrum": {"periods": [0.05, 0.1, 0.5, 1.0, 3.0],
                      "values": [0.2, 0.3, 0.3, 0.2, 0.1], "unit": "g"},
  "t1": 0.1, "t2": 2.0, "s_min": 0.5, "s_max": 3.0,
  "top_n": 7, "quantity": "psa", "rotd_band": "rotd50",
  "bounds_policy": "reject"
}
```

- 误差按**组合谱**（RotD50 或 RotD100）与设计谱在同一周期网格上比较，
  最优缩放仍是对数空间几何平均闭式解，越限处理（reject/clamp）、
  排序（mse 升序、同 mse 按 group_id）、前 N 组算术/几何平均谱、设计谱
  控制点逐点比值——口径与单条匹配（§6.5）完全一致；
- **同一组两条水平分量共用一个缩放系数**；`ranking[].members_scaled`
  给出每组缩放后两条分量各自的谱（可直接用于双向时程成对缩放）。

### 12.4 组合谱缓存复用（三处取舍之三）

组合谱算过一次后，后续组匹配作业（及重复提交的组合谱作业）**复用**
结果：每个组每套参数的结果以内容寻址键写入 `rot_spectra` 表，键 =
`组 id + 对齐后两条序列字节 + dt + 全部积分参数 + CALC_VERSION` 的
哈希。组一旦建立即不可变（成员与对齐结果都进了组 id 与组行；要换
成员只能另建新组），因此：

- 组成员变了 → 对齐序列字节变 → 组 id 与缓存键都变，不会命中间结果；
- 周期/阻尼/方法/基线/加密策略/`n_angles` 任一参数变了 → 键变；
- 组合计算的数值逻辑升级 → 升 `app/rotd.py:CALC_VERSION` → 旧键全部
  失配，自动重算。

代价是缓存表会累积历史结果（当前不做淘汰；结果为 JSON、单条百 KB
量级，200 组上限下占用可接受，后续可加 LRU/TTL）。作业结果内
`cache_hit` 与 `summary.cache_hits` 标明是否走了缓存。

---

## 13. 升级兼容

挂载卷里旧版本的库文件可直接用新版镜像启动：`Storage` 打开旧库时做
**只增不改**的增量迁移——新建 `component_groups`、`rot_spectra` 表，
放宽 `jobs.type` 的 CHECK（连同被外键引用的 `job_items` 一起重建并
原样拷数据），给 `job_items` 加可空 `group_id`。老记录、老作业（含
执行中→interrupted、排队中→继续执行）、老逐条结果字段与返回结构
全部不变；单条记录上传、`/api/jobs/spectrum`、`/api/jobs/match` 的
路径、默认值与返回结构保持原样，本次升级**只新增字段与端点**。
该路径由 `tests/test_upgrade.py` 用手工构造的旧版本结构库文件验证。
