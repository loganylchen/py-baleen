# Eventalign Pipeline — Implementation Plan

## Overview

端到端 nanopore 信号分析流水线：

**输入**: FASTQ + BAM (transcriptome-aligned) + BLOW5 (原始信号) × 2 套 (native + IVT)
**输出**: 每个 contig 每个 position 的 pairwise DTW distance matrix

核心设计：**generator-based lazy processing** — 逐个 contig 生成临时 BAM、跑 eventalign、提取信号、计算 DTW、清理临时文件。不一次性生成所有中间文件。

---

## Module 结构

```
baleen/
├── __init__.py
├── _cuda_dtw/                # (已有) DTW 计算
└── eventalign/
    ├── __init__.py            # 公共 API 导出
    ├── _f5c.py                # f5c 软件检测、版本、index、eventalign 调用
    ├── _bam.py                # pysam BAM 操作: 拆分、depth 计算、filtering
    ├── _signal.py             # eventalign TSV 解析 → 位点信号提取
    └── _pipeline.py           # 主 pipeline 编排逻辑
```

---

## Step 1: `_f5c.py` — f5c 工具封装

所有与 f5c CLI 的交互封装在此模块。

### 函数

| 函数 | 说明 |
|---|---|
| `check_f5c() → str` | 检查 `f5c` 是否在 PATH 中，运行 `f5c --version`，返回版本号字符串（如 `"1.6"`）。不存在则 raise `RuntimeError` |
| `get_f5c_version() → tuple[int, ...]` | 解析版本号为 tuple `(1, 6)`，方便做版本兼容检查 |
| `is_indexed(fastq) → bool` | 检查 `<fastq>.index.readdb` 是否存在且非空 |
| `index_fastq_blow5(fastq, blow5) → None` | 运行 `f5c index --slow5 <blow5> <fastq>`。先调用 `is_indexed()` 检查，已存在则跳过 |
| `run_eventalign(bam, ref_fasta, fastq, blow5, output_tsv, extra_args=None) → Path` | 运行 `f5c eventalign -b <bam> -g <ref> -r <fastq> --slow5 <blow5> --samples --scale-events > <output_tsv>` |

### 细节

- 所有 subprocess 调用使用 `subprocess.run(check=True, capture_output=True)`，失败时抛出带 stderr 信息的异常
- f5c version 解析: `f5c --version` 输出类似 `f5c v1.6`，用正则 `r"v?(\d+\.\d+.*)"` 提取
- Index 产物: `f5c index` 会创建 `<fastq>.index`, `<fastq>.index.fai`, `<fastq>.index.gzi`, `<fastq>.index.readdb`
- BLOW5 index: 检查 `<blow5>.idx` 是否存在，如不存在则需要 `slow5tools index <blow5>`

### 待确认

- [ ] `f5c index` 是否自动处理 blow5 的 `.idx` index？还是需要单独 `slow5tools index`？
- [ ] eventalign 是否需要 `--signal-index` 参数？需要
- [ ] 你需要传什么 `extra_args`？比如 `--rna`, `--kmer-model` 等？需要，而且可能还需要比如针对RNA002和RNA004不同的参数，并且还要保留readname，这样我就可以进行快速识别，read的类型

---

## Step 2: `_bam.py` — BAM 操作

pysam 封装：BAM 验证、contig 统计、拆分 generator、过滤逻辑。

### 数据结构

```python
@dataclass
class ContigStats:
    contig: str
    mapped_reads: int
    mean_depth: float
```

### 函数

| 函数 | 说明 |
|---|---|
| `validate_bam(bam_path) → None` | 检查 BAM 是否存在、sorted、indexed（`.bai` 存在）。不满足则 raise `ValueError` |
| `get_contig_stats(bam_path) → dict[str, ContigStats]` | 用 `get_index_statistics()` 获取 mapped reads 数，用 `count_coverage()` 计算 mean depth |
| `filter_contigs(native_stats, ivt_stats, min_depth=15) → list[str]` | 返回 native 和 IVT **同时满足** `mean_depth >= min_depth` 的 contig 列表 |
| `split_bam_contig(bam_path, contig, output_dir) → Path` | 提取指定 contig 的 reads 写入临时 BAM + 自动 `pysam.index()`，返回临时 BAM 路径 |

### Generator 设计

```python
def iter_contig_bams(bam_path, contigs):
    """
    Generator: 逐个 yield 指定 contig 的临时 BAM 路径。
    每次 yield 后，caller 处理完毕，控制权返回时清理临时文件。
    """
    with pysam.AlignmentFile(bam_path, "rb") as samfile:
        for contig in contigs:
            tmp = tempfile.NamedTemporaryFile(suffix=f"_{contig}.bam", delete=False)
            tmp_path = Path(tmp.name)
            tmp.close()
            
            with pysam.AlignmentFile(str(tmp_path), "wb", template=samfile) as out:
                for read in samfile.fetch(contig):
                    out.write(read)
            pysam.index(str(tmp_path))
            
            try:
                yield contig, tmp_path
            finally:
                # yield 返回后清理
                tmp_path.unlink(missing_ok=True)
                Path(str(tmp_path) + ".bai").unlink(missing_ok=True)
```

### Filtering 逻辑

- 输入: native BAM 的 contig stats + IVT BAM 的 contig stats
- 条件: **两者都**满足 `mean_depth >= min_depth`（默认 15）
- 某 contig 只在一方满足 → 跳过
- 某 contig 只在一个 BAM 中出现 → 跳过
- 还要加入一下filtering的参数，比如是不是primary alignment之类的

---

## Step 3: `_signal.py` — Eventalign 解析 & 信号提取

解析 f5c eventalign TSV 输出，提取每个位点覆盖的所有 reads 的信号。

### Eventalign TSV 格式 (with `--samples`)

```
contig  position  reference_kmer  read_name  strand  event_index  event_level_mean  event_stdv  event_duration  model_predict  model_stdv  samples
ENST... 100       AACCC           read_001   t       5            120.5             2.3         0.005           118.2          3.1         116.89,118.60,120.15,...
```

### 数据结构

```python
@dataclass
class PositionSignals:
    contig: str
    position: int
    reference_kmer: str
    read_signals: dict[str, np.ndarray]  # read_name → 该 read 在此 position 的完整信号
```

### 函数

| 函数 | 说明 |
|---|---|
| `parse_eventalign(tsv_path) → Generator[dict]` | 逐行解析 TSV，yield 结构化 dict |
| `group_signals_by_position(tsv_path) → dict[int, PositionSignals]` | 按 position 分组，合并同一 read 在同一 position 的多个 event |
| `extract_position_signals(position_signals: PositionSignals) → list[np.ndarray]` | 从 PositionSignals 提取信号列表，用于 DTW 输入 |

### 信号合并策略

同一个 read 在同一个 position 可能产生多个 event row（多行），需要合并：
- **策略**: 将同一 (read_name, position) 的所有 event 的 `samples` 列拼接为一个连续 `np.ndarray`
- 最终每个 read 在每个 position 得到一个变长信号 array

### 待确认

- [ ] 信号粒度: 使用 `--samples` 的 raw 信号点，还是只用 `event_level_mean`？用samples
- [ ] 同一 read 同一 position 多个 event 的 samples 是直接 concatenate，还是有其他合并方式？，直接concatenate，但是要注意，就是RNA的信号和sequence是反的。所以要注意signal的index信息，

---

## Step 4: `_pipeline.py` — 主流水线

编排整个流程。

### 公共 API

```python
def run_pipeline(
    native_bam: Path,
    native_fastq: Path,
    native_blow5: Path,
    ivt_bam: Path,
    ivt_fastq: Path,
    ivt_blow5: Path,
    ref_fasta: Path,
    *,
    min_depth: int = 15,
    use_cuda: Optional[bool] = None,
    use_open_start: bool = False,
    use_open_end: bool = False,
    output_dir: Optional[Path] = None,
    cleanup_temp: bool = True,
) -> dict[str, ContigResult]:
```

> **注**: 这里假设 native 和 IVT 各有独立的 FASTQ + BLOW5。
> 如果共用同一个 FASTQ + BLOW5，参数可以简化。**待确认。**

### 流程

```
1. 检查环境
   ├─ f5c_version = check_f5c()
   └─ 存储到 metadata

2. 检查 / 执行 indexing
   ├─ native: is_indexed(native_fastq) ? skip : index_fastq_blow5(native_fastq, native_blow5)
   ├─ ivt:    is_indexed(ivt_fastq)    ? skip : index_fastq_blow5(ivt_fastq, ivt_blow5)
   ├─ native blow5: <native_blow5>.idx 存在? skip : slow5tools index
   └─ ivt blow5:    <ivt_blow5>.idx 存在?    skip : slow5tools index

3. 获取两个 BAM 的 contig 统计
   ├─ validate_bam(native_bam)
   ├─ validate_bam(ivt_bam)
   ├─ native_stats = get_contig_stats(native_bam)
   └─ ivt_stats = get_contig_stats(ivt_bam)

4. 过滤 contigs
   └─ valid_contigs = filter_contigs(native_stats, ivt_stats, min_depth)
   └─ 如果 valid_contigs 为空 → 提前返回空结果 + warning

5. 逐 contig 处理 (generator 模式):
   results = {}
   for contig in valid_contigs:

     5a. 拆分 BAM
         ├─ native_contig_bam = split_bam_contig(native_bam, contig, tmp_dir)
         └─ ivt_contig_bam = split_bam_contig(ivt_bam, contig, tmp_dir)

     5b. f5c eventalign
         ├─ native_ea = run_eventalign(native_contig_bam, ref_fasta, native_fastq, native_blow5, tmp_native_ea.tsv)
         └─ ivt_ea = run_eventalign(ivt_contig_bam, ref_fasta, ivt_fastq, ivt_blow5, tmp_ivt_ea.tsv)

     5c. 解析 eventalign → 按 position 分组
         ├─ native_pos_signals = group_signals_by_position(native_ea)
         └─ ivt_pos_signals = group_signals_by_position(ivt_ea)

     5d. 对于每个 position (取两组共有的 positions):
         ├─ native_signals: list[np.ndarray] = 提取该 position 所有 read 信号
         ├─ ivt_signals: list[np.ndarray] = 提取该 position 所有 read 信号
         ├─ all_signals = native_signals + ivt_signals   (或者分开计算？待确认)
         └─ distance_matrix = dtw_pairwise(all_signals, use_cuda=use_cuda, ...)

     5e. 存储结果
         └─ results[contig] = ContigResult(...)

     5f. 清理临时文件
         └─ 删除 contig BAM + eventalign TSV

6. 返回 results
```

### 结果数据结构

```python
@dataclass
class ContigResult:
    contig: str
    native_depth: float
    ivt_depth: float
    positions: dict[int, PositionResult]

@dataclass
class PositionResult:
    position: int
    reference_kmer: str
    n_native_reads: int
    n_ivt_reads: int
    native_signals: list[np.ndarray]    # 可选保留，或只保留 matrix
    ivt_signals: list[np.ndarray]       # 可选保留
    distance_matrix: np.ndarray         # (n_native + n_ivt, n_native + n_ivt) DTW 矩阵
    # distance_matrix 的前 n_native 行/列 = native reads
    # 后 n_ivt 行/列 = ivt reads
    # 这样可以从 matrix 中提取:
    #   - native vs native (左上角)
    #   - ivt vs ivt (右下角)
    #   - native vs ivt (右上角 / 左下角)

@dataclass
class PipelineMetadata:
    f5c_version: str
    min_depth: int
    use_cuda: Optional[bool]
    n_contigs_total: int
    n_contigs_passed_filter: int
    n_contigs_skipped: int
```

---

## Step 5: Dependencies

### 新增依赖

| Package | 用途 | 必须? |
|---|---|---|
| `pysam` | BAM 读写、拆分、depth 计算 | 是 |

### 不需要的依赖

- **pyslow5**: Python 端不解析 blow5，f5c 直接读取
- **biopython**: Python 端不解析 FASTQ，f5c 直接使用
- **pandas**: eventalign 解析用 csv module + numpy 足够

### 外部工具依赖 (非 pip)

| 工具 | 用途 | 检测方式 |
|---|---|---|
| `f5c` | eventalign + index | `f5c --version` |
| `slow5tools` | blow5 indexing (如果 f5c index 不自动处理) | `slow5tools --version` |

---

## Step 6: 测试策略 (TDD)

每个模块先写测试，再写实现。

| 测试文件 | 测试内容 | 策略 |
|---|---|---|
| `tests/test_f5c.py` | f5c 检测、版本解析、index 检查逻辑、eventalign 命令构造 | Mock `subprocess.run`，不依赖真实 f5c |
| `tests/test_bam.py` | BAM 验证、contig stats、拆分 generator、filtering 逻辑 | 用 pysam 在测试中构造小型 BAM 文件 |
| `tests/test_signal.py` | eventalign TSV 解析、信号分组、合并 | 手工构造 TSV fixture 文件 |
| `tests/test_pipeline.py` | 端到端 pipeline 编排逻辑 | Mock f5c subprocess + 小型真实 BAM |

### 测试风格 (跟现有一致)

- pytest + test classes
- 确定性 RNG: `np.random.default_rng(seed=42)`
- 明确的 error message 断言
- 临时文件用 `tmp_path` fixture

---

## Step 7: 实现顺序

```
1. _f5c.py     + tests/test_f5c.py       (最独立，纯 subprocess 封装)
2. _bam.py     + tests/test_bam.py       (依赖 pysam)
3. _signal.py  + tests/test_signal.py    (纯文件解析)
4. _pipeline.py + tests/test_pipeline.py (组装以上模块)
5. eventalign/__init__.py               (公共 API 导出)
6. 更新 pyproject.toml + setup.py        (添加 pysam 依赖)
```

---

## Open Questions

以下问题需要你确认后才能开始实现：

### Q1: 输入文件关系
Native 和 IVT 是否各有**独立的** FASTQ + BLOW5？
- A: 是，各自独立 → pipeline 接受 6 个文件 (native_bam + native_fastq + native_blow5 + ivt_bam + ivt_fastq + ivt_blow5)
- B: 共用同一个 FASTQ + BLOW5，只是 BAM 不同 → pipeline 接受 4 个文件

- 选择A，最好还要输入reference，这样可以根据reference来判断大家是不是在genrator同一个contig，如果都通过filter，那就继续，如果有一个没有通过filter，就skip，但是需要记录，这条contig为什么被filtering了。

### Q2: DTW 比较对象
同一个 position 上的信号，DTW 怎么比较？
- A: **All pairwise** — native + ivt 所有 reads 合在一起做 full pairwise DTW → 一个大矩阵
- B: **Cross only** — 只做 native reads vs ivt reads 的 cross-pairwise
- C: **Three matrices** — native-native + ivt-ivt + native-ivt 三个子矩阵

- 选择A，一个大矩阵

### Q3: 信号粒度
- A: 使用 `--samples` 的 raw 信号点 (每个 event 有多个采样点)
- B: 只使用 `event_level_mean` (每个 event 一个值)

选择A

### Q4: Reference FASTA
`-g` 参数传的是什么？
- A: Transcriptome FASTA (因为 BAM 是 aligned to transcriptome)
- B: Genome FASTA

选择A

### Q5: 信号合并
同一个 read 在同一个 position 有多个 event 时：
- A: Concatenate 所有 event 的 samples
- B: 只取第一个 event
- C: 其他方式

选择A

### Q6: 输出格式
Pipeline 结果怎么保存？
- A: 只返回 Python 对象 (dict of dataclasses)，由 caller 决定如何保存
- B: 自动保存到文件 (HDF5 / pickle / numpy)
- C: 两者都支持 (默认返回对象，可选保存到文件)

选择C

### Q7: Logging
- A: 继续用 `print`（跟现有 DTW 模块一致）
- B: 使用 `logging` module（pipeline 更需要结构化日志）
- C: 两者并存（DTW 保持 print，pipeline 用 logging）

选择B
