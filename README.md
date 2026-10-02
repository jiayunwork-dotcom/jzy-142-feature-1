# 弹性反应谱计算与谱匹配作业服务

把一批强震加速度记录提交上来，后台用 Newmark-β 逐步积分计算每条记录的
弹性反应谱（SD / SV / SA / PSV / PSA），再按调用方给出的设计谱做
周期区间内的对数误差最优缩放与匹配筛选，输出排序、前 N 条平均谱及
逐点比值。上传、计算、匹配都是异步作业，可查进度 / 部分结果 / 取消；
记录、作业状态与结果全部落 SQLite，进程重启后已完成结果照常可读，
执行中的作业标记为中断。

**新版支持水平分量组（RotD）**：把同一台站同一次地震的两条水平分量
作为一个一等对象，后台计算方向无关组合谱 RotD50 / RotD100（含
RotD100 方位角），匹配作业以组为单位、两条分量共用一个缩放系数，
误差按 RotD100 与设计谱比较（见 §11–§13）。竖向分量可挂在组里保留，
不参与水平组合。

- Python 3.12 + Flask + NumPy
- 积分器、反应谱、RotD 组合、匹配优化均为自实现（仅线性代数借助 NumPy）
- 容器：`python:3.12-slim`，gunicorn 启动，SQLite 在挂载卷

---

## 1. 目录结构

```
app/
  errors.py       # 领域异常（逐条可读错误；新增 GroupError）
  parsing.py      # 记录文本解析（两列 / 单列+步长、文件头元数据、首点时刻）
  baseline.py     # 基线校正（none/mean/linear）与速度位移积分
  integrator.py   # Newmark-β 积分器（峰值版 + RotD 用响应时程版）
  spectrum.py     # 单条弹性反应谱 SD/SV/SA/PSV/PSA、零周期点、PGA/PGV/PGD
  groups.py       # 水平分量组：单位统一、对齐（并集/重采样/补零）、配对建议
  rotd.py         # RotD50/RotD100 组合谱：方位角投影、分块组织、误差口径、指纹
  group_jobs.py   # rotd / match_group 两类组作业执行逻辑与结果缓存复用
  matching.py     # 设计谱对数插值、最优缩放、越限剔除、排序、平均谱比值
                  # （单条记录与分量组同一套口径）
  jobs.py         # spectrum / match 两类单条作业的执行逻辑
  scheduler.py    # 单工作线程后台调度、崩溃恢复
  storage.py      # SQLite：records/jobs/job_items/component_groups/rotd_cache
                  # 与旧版本（v0）库的在线迁移
  api/
    records.py    # 记录上传与查询（路径/默认值/返回结构不变）
    jobs.py       # 作业创建 / 查询 / 取消（新增 rotd、match-group 两个端点）
    groups.py     # 分量组建组、查询、配对建议
    helpers.py
  __init__.py     # Flask 应用工厂
wsgi.py           # gunicorn 入口
tests/            # pytest（新增 rotd/groups/group_api/旧库升级用例）
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

## 9. 水平分量组

分量组（component group）是一等对象：由**同一台站同一次地震的两条
水平分量**构成；竖向分量可以挂在组里保留（`vertical_record_id`），
但不参与任何水平组合。组落库（`component_groups` 表），重启后仍在。

### 9.1 配对建议（只建议，不自动建）

```bash
curl -s -X POST http://localhost:8000/api/groups/suggestions \
  -H 'Content-Type: application/json' \
  -d '{"record_ids": ["rec_...", "rec_..."]}'   # record_ids 可省略，对全库建议
```

服务读取上传时文件头抽取的 `station / event / component / channel`
元数据（# 头里写的 N/E/Z、north/east、分量 30 deg 等），按
(station, event) 聚簇：

- 两条水平分量方位角差在 90°±12° 内（或明确标 N/E、X/Y）：
  `confidence="orthogonal"`，给出可读理由；
- 同簇两条水平但方向无法确认正交：`confidence="candidate"`，
  理由里写明缺少方向信息，由调用方人工确认；
- 竖向分量、找不到配对的分量进入 `unpaired`。

**建议是只读的，服务绝不据此自动建组**；必须再由调用方显式点名建组。

### 9.2 显式建组

```bash
curl -s -X POST http://localhost:8000/api/groups \
  -H 'Content-Type: application/json' -d '{
    "horizontal1": "rec_aaa", "horizontal2": "rec_bbb",
    "vertical_record_id": "rec_ccc",
    "name": "ST1-EV1 水平分量对"
}'
```

组 ID 内容寻址：两条水平分量**无序**（交换 h1/h2 是同一个组）、
竖向分量参与哈希。成员变化（哪怕换一条分量）必然产生新组 ID——这是
组合谱结果缓存安全复用的前提。

建组返回（及 `GET /api/groups/<id>`）里有完整的 `alignment` 段，
写明实际采用的每一步处理（见 §11.2）。建组失败（成员不可用、对齐后
超 20 万点等）返回 400 和可读原因，不产生半成品组。

接口一览：

| 接口 | 说明 |
|---|---|
| `POST /api/groups/suggestions` | 配对建议（只读） |
| `POST /api/groups` | 显式点名两条水平分量建组（可挂竖向） |
| `GET /api/groups` | 组列表 |
| `GET /api/groups/<id>` | 组详情（含 alignment 说明） |

---

## 10. 组合谱作业与组匹配作业

### 10.1 RotD 组合谱作业

`POST /api/jobs/rotd`

```json
{
  "group_ids": ["grp_...", "grp_..."],
  "periods": [0.0, 0.02, 0.05, 0.1, 0.5, 1.0, 3.0, 6.0],
  "dampings": [0.02, 0.05],
  "method": "average_acceleration",
  "instability_policy": "refine",
  "baseline_mode": "mean",
  "n_angles": 180
}
```

参数默认值与单条谱作业完全一致；`n_angles` 默认 **180**（[0°,180°)
上 1° 网格），须为偶数、范围 [8, 7200]。作业机制沿用现有框架：后台
串行执行、`GET /api/jobs/<id>` 查进度、`/items` 看逐组部分结果、
可取消；某一组算不出来只把该组标记为 `skipped/error` 并注明原因，
作业整体仍 `completed`，`result.summary` 给出
done/error/skipped/**cache_hits** 计数。一批最多 200 组。

每组、每个阻尼比的结果含：

- `rotd50` / `rotd100`：位移 `sd`、相对速度 `sv`、绝对加速度 `sa`、
  伪速度 `psv`、伪加速度 `psa`（位移/伪加速度/绝对加速度三类量齐全）；
- `rotd100.angle_deg`：逐周期取得 RotD100 的方位角（0–180°）；
- `components[2]`：两条原始水平分量各自的 sd/sv/sa/psv/psa（及 g 制）；
- `geometric_mean`：两条分量谱的几何平均（对照用，业内常用的 GMRotD
  前身）；
- `ground_motion`：两条分量的 PGA/PGV/PGD 与水平面 RotD100/RotD50 PGA；
- `method / dt_used / refined / refine_factor / tail_steps / baseline`：
  与单条谱一致的数值方法元数据。

### 10.2 以组为单位的匹配作业

`POST /api/jobs/match-group`，参数在 §4.2 基础上把 `record_ids` 换成
`group_ids`：

```json
{
  "group_ids": ["grp_..."],
  "periods": [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0],
  "dampings": [0.05], "target_damping": 0.05, "n_angles": 180,
  "design_spectrum": {"periods": [0.01, 0.1, 0.5, 1.0, 6.0],
                      "values": [0.20, 0.60, 0.60, 0.35, 0.10], "unit": "g"},
  "t1": 0.1, "t2": 2.0, "s_min": 0.5, "s_max": 3.0,
  "top_n": 7, "quantity": "psa", "bounds_policy": "reject"
}
```

口径与单条匹配严格一致（同一实现），区别只有：

- 候选单位是**组**；同一组两条分量**共用一个缩放系数**；
- 记录谱取该组目标阻尼下的 **RotD100**（`matched_spectrum="rotd100"`），
  越限处理（reject/clamp）、对数误差闭式最优、按 mse 与组 id 排序、
  前 N 组缩放后谱的算术/几何平均、设计谱控制点上的逐点比值都不变；
- 输出额外给出 `scaled_groups`：每个入榜组缩放后的 RotD100/RotD50，
  以及**缩放后两条分量各自的谱**（`components_scaled`，
  sd/sv/sa/psv/psa 全套），方便复核两条分量是否被同一系数合理缩放；
- 排序结果键名为 `group_id`（单条匹配仍是 `record_id`，字段不变）。

整组乘 k 时 RotD 谱乘 k（线性体系），匹配闭式解给出缩放系数
`s* → s*/k`，与单条记录的缩放互易关系一致（测试逐条验证）。

---

## 11. 三处关键取舍（理由与代价）

### 11.1 方位角离散与计算组织

**做法**。线性 SDOF 体系满足叠加原理：对两条正交水平分量的响应
（相对位移 u、相对速度 v）各积分**一次**，投影到任意方位角只需

```
u_θ(t) = u1(t)·cosθ + u2(t)·sinθ
```

取 `u_θ` 时程峰值得到该角度的谱值。由于谱对 θ 有 π 周期性
（|响应| 峰值在 θ+π 相同），方位角只扫 **[0°,180°)**，默认 180 点、
步长 **Δ=1°**。RotD100 为全部角度峰值的最大（记录其方位角），
RotD50 为这些峰值样本的中位（偶数点取中间两个的算术平均——NumPy
partition 精确实现，不依赖插值近似）。

**精度 / 误差上限**：

- RotD100 只可能被离散化**低估**。逐角度峰值可写成椭圆型
  `S(θ)² = A·cos²θ + B·sin²θ + 2C·sinθcosθ`，曲率有界，故网格相对
  误差 ≤ **Δ²/8**；Δ=1° 时上界 **3.9×10⁻⁵**，10 组随机信号对实测
  最坏 3.7×10⁻⁵，服务声明上界 **5×10⁻⁵**。
- RotD50 是样本中位数，分箱统计是**一阶 O(Δ)**（不是二阶）：
  2° 网格实测最坏 2.6×10⁻³，1° 网格约 9×10⁻⁴，0.5° 后进入噪声平台。
  服务声明 RotD50 上界 **2×10⁻³**（1° 网格）。
- 所有「受方位角离散影响」的测试比较（旋转非整数网格角）采用
  RotD100 **1×10⁻⁴**、RotD50 **3×10⁻³** 的相对容差，与本声明一致，
  不为测试另放宽；网格整数角旋转（30°、117°）、交换、取反是角度集合
  的纯重排，按 1×10⁻¹² 严格容差验证。需要更高精度时可调
  `n_angles`（最大 7200，0.025°）。

**代价**。另一种组织是「逐个角度重新积分」，精度定义相同但积分
次数随角度数成倍增长（默认 180 倍）。本实现只积分一次、把
「时间步 × 周期 × 方位角」的投影峰值统计作为额外成本，并两级分块
（见 §14），代价是要保留两条分量的响应时程中间量；为把内存压在上限
内，周期维分块（每块 ≤12 个周期），代价是积分 Python 时间步循环
按 ⌈周期数/12⌉ 倍重复（默认 100 周期 → 9 块）。实测单组典型记录
（3000 点、8 周期、180 角、5% 阻尼）约 0.16 s。

### 11.2 两条分量对不齐时的策略

单位、步长、时长/点数、起始时刻分别处理，实际操作全部写进组信息
`alignment` 段：

1. **单位不同先统一**：g 一律按 9.80665 换算到 m/s² 后再组合；
2. **起始时刻**：时间–加速度两列格式保留首点时间（新增
   `metadata.time_start`，纯新增字段），单列/序号格式起始按 0。
   两条时间轴取**并集**，各自覆盖区间之外的采样补零；
3. **步长不一致**：**线性插值重采样到较细步长**（细步长不是粗步长
   整数分点时粗序列会被插值；强震记录带限、且线性加速度法本身假设
   区间内加速度线性，与之一致），不做任意比例重采样到新步长；
4. **时长/点数不同**：取并集自然解决，缺口补零。

**为什么不截齐**：截掉较长记录的真实地震动会系统性改变谱值（尤其
长周期段），且调用方无从察觉；补零在物理上等价于「该方向在那段时间
无记录」，是可逆、可声明的处理。**唯一拒绝情形**：对齐后总点数超过
20 万上限——此时报错并给出起始时刻、采用步长、点数和处理建议，由
调用方截短或换更粗步长的分量，服务不擅自截断。每条分量的
`resampled / zero_padded / pad_start / pad_end` 与人类可读的
`notes` 都在组信息中写明。

### 11.3 组合谱结果是否复用、如何防旧结果

**复用**。rotd 与 match-group 作业计算前先查 `rotd_cache`，命中则
直接采用并在作业项标 `result_source="cache"`、summary 计
`cache_hits`；否则计算并回写。同一组同一套参数重复提交因此逐位
相同（测试验证 JSON 逐位一致），匹配作业也复用先前 rotd 作业的
结果，避免重复积分。

**防旧结果靠内容指纹**。缓存键 `rotd_fingerprint` 覆盖：

- 组**内容键**（两条水平分量无序 + 竖向分量的 SHA-256）；
- 全部积分参数：周期列表、阻尼列表、method、instability_policy、
  baseline_mode、n_angles；
- 算法版本号 `ROTD_ALGORITHM_VERSION`（组合算法若将来调整，旧缓存
  一律失效，强制重算）。

组本身**不可变**（没有「编辑组成员」的接口——换成员得到的是新组 ID
和新内容键），所以「组成员变了拿到旧结果」在结构上不可能发生；
任何一个积分参数变化指纹即变。竖向分量不参与组合，但其身份仍进入
组内容键（挂不同竖向记录是不同组，结果不串用）。

---

## 12. 容量与性能（分量组）

- 上限沿用单条口径：一批最多 **200 组**、对齐后每组最多
  **200 000** 点（上传与建组两处校验）、默认 **100** 个周期点；
- 两级分块保证不把容器内存撑爆（20 万点 × 100 周期的极端组合）：
  - 周期分块 `PERIOD_BLOCK=8`：同时只保留两条分量 8 个周期的
    u/v 响应时程，约 4 × 200001 × 8 × 8 B ≈ **51 MB**；
  - 时间行分块 `ROW_BLOCK=4096`：方位角投影写成
    (行×周期×2)×(2×方位角) 的批量矩阵乘，单个
    (行×周期×角) 缓冲依次复用，约 4096 × 8 × 180 × 8 B
    ≈ **47 MB**；
  - 加上地面加速度、基线序列、各统计量与 NumPy/解释器基线，极端组合
    **实测峰值 RSS 约 230–250 MB**（20 万点 × 100 周期 × 180 角，
    默认 python:3.12-slim 无额外内存配置；典型数千点记录峰值仅数
    MB），耗时约 80 s 量级（后台批处理，单工作线程）。分块常数集中在
    `app/rotd.py` 顶部，内存更紧时可调小，代价是积分循环按块数倍增。
- 不缓存逐时程中间量到磁盘：缓存只存最终谱（JSON），重启后结果
  可读但需重算时按指纹重新积分。

---

## 13. 升级兼容（旧版本库直接启动）

升级后用挂载卷里的**原库文件**直接启动即可，启动时自动做一次性
在线迁移（`storage.py::_migrate_v0_to_v1`，以 `PRAGMA user_version`
标记）：

- `records` 不动；`jobs` 重建以把 type 的 CHECK 扩成
  spectrum/match/**rotd/match_group**（整行连同时间戳拷回）；
- `job_items` 增加可空 `group_id` 列（老数据为 NULL，重建表拷回，
  迁移期间临时关闭外键以避开 SQLite RENAME 改写外键引用导致的
  级联删除，并在迁移后跑 `foreign_key_check` 校验）；
- 新增 `component_groups` / `rotd_cache` 两张表。

老作业照常能查、字段与返回结构不变（`group_id` 是新增字段）；
单条记录上传、`/api/jobs/spectrum`、`/api/jobs/match` 的路径、
默认值、返回结构全部保持原样，本次改动**只新增字段/端点**。
`tests/test_upgrade_legacy_db.py` 用按旧版本原始建表语句手工构造的
库文件验证整段升级过程。

---

## 14. 测试覆盖（pytest，共 130+ 用例）

需求中列出的关系均有**独立测试**逐条确认：

**单条记录（原有）**：
- 记录整体乘 k：所有谱值乘 k、匹配缩放系数除以 k
  （`test_scaling_linearity*`、`test_scaled_record_scales_inverse`）；
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

**分量组与 RotD（新增，`test_rotd.py` / `test_groups.py` /
`test_group_api.py`）**：
- 两条分量一起绕竖轴转 **30° 和 117°**（均为 1° 网格的整数倍，
  角度集合纯重排）后作为新组，RotD50/RotD100 与原组逐位一致
  （1e-12）；转非网格角 30.5° 时偏差不超过 §11.1 声明的
  RotD100 1e-4 / RotD50 3e-3 容差，并有 1° vs 0.1° 网格收敛测试
  证实声明上界 5e-5；
- 交换两条分量、把其中一条整体取反，组合谱不变；
- 一条分量全为零：RotD100 等于另一条分量单独的谱（RotD50 等于
  √2/2 倍单条谱——逐角度峰值 |cosθ| 的中位，测试中写明该定义）；
- 两条分量完全相同：RotD100 是单条谱的 √2 倍（RotD50 不是 √2，
  各角度峰值在 45° 取最大，测试注释了这一物理事实）；
- 任何周期 RotD100 ≥ RotD50，且 ≥ 两条原始分量谱的较大者，
  SA/PSA 均验证；
- 整组乘 k，组合谱（三类量）随之乘 k、RotD100 方位角不变；组匹配
  缩放系数变为原来的 1/k（端到端 HTTP 验证）；
- 同一组同一套参数重复提交，组合结果 JSON **逐位相同**（缓存命中、
  缓存未命中两条路径都验证）；缓存指纹对组成员/任一积分参数/算法
  版本敏感；
- 单位统一（g↔m/s²）、步长不一致重采样、起始时刻错开补零、并集超
  20 万点拒绝（可读原因）、同一条记录不能当两条分量、竖向分量挂接；
- 配对建议：N/E 与角度（10°/100°）正交对、方向缺失给候选、Z 进
  unpaired、建议接口只读不建组；
- 组匹配：共用系数、越限 reject/clamp、排序、平均谱、逐点比值、
  每组缩放后两条分量各自的谱；某组缺失/失败只标记该组；批量上限、
  n_angles 校验、取消、组与结果跨重启可读。
- **旧版本库升级启动**（`test_upgrade_legacy_db.py`）：用旧版本
  原始建表语句构造含已完成/运行中作业的老库，新版直接打开后老数据
  与字段不变、running 作业恢复为 interrupted、新功能可正常建组跑
  rotd 作业、二次启动不重复迁移。

## 15. 容量与性能

- 批量 ≤ 200 条（组）、单条（对齐后每组）≤ 200 000 点（上传、
  建组与建作业三处都校验）；
- 典型强震记录（数千点、100 周期点、5% 阻尼）单条亚秒级；
- 20 万点 × 100 周期的极端组合每步是 100 维向量运算，主要成本为
  Python 时间步循环，属后台批处理可接受范围；积分热点全部在 NumPy。
  组的极端组合因周期分块会按 ⌈周期数/12⌉ 倍重复积分循环，内存代价
  与实测峰值见 §12。
