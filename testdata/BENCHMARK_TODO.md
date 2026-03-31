# Benchmark TODO: 验证低 Stoichiometry 修饰检测改进

## 背景

修改了 `_hierarchical.py` 的 V2 scoring：用 contig-level global alternative 替代 per-position EM，解决低 stoichiometry（<30%）下 AUPRC 崩塌的问题。需要用 ecoliRNA100 testdata 验证效果。

## 前置条件

- [x] testdata/ 下 0.0–1.0 共 11 个 stoichiometry 级别的 blow5 + fastq 已就绪
- [x] `testdata/ref.fa` 参考基因组已就绪
- [x] `testdata/known_modifications.tsv` ground truth（36 个已知 E. coli rRNA 修饰位点）已就绪
- [x] `testdata/prepare_bams.sh` BAM 生成脚本已就绪
- [ ] BAM 文件已生成（需要 minimap2 + samtools）
- [ ] f5c 已安装且在 PATH 上

---

## Task 1: 写 benchmark 脚本（跑所有 stoichiometry 级别）

写一个脚本 `testdata/run_benchmark.sh`（或 Python），对 testdata/ 下 0.0–1.0 共 11 个 stoichiometry 级别分别跑 `baleen run`：

- native: `testdata/{stoich}/data/native_1/native_1.bam` + 对应 fastq/blow5
- IVT control: `testdata/{stoich}/data/control_1/control_1.bam` + 对应 fastq/blow5
- ref: `testdata/ref.fa`
- 输出到 `testdata/{stoich}/output/`（新 scoring）和 `testdata/{stoich}/output_legacy/`（旧 scoring）
- 每个级别跑两次：一次默认（新 contig-level scoring），一次 `--legacy-scoring`
- 用 `--no-cuda`（本地无 GPU）
- 跑完后每个 output 目录应有 `site_results.tsv`

## Task 2: 写 AUPRC/AUROC 评估脚本

写一个 Python 脚本 `testdata/evaluate_benchmark.py`，读取所有 stoichiometry 级别的 `site_results.tsv`，与 `testdata/known_modifications.tsv` ground truth 对比：

1. 对每个 stoichiometry 级别：
   - 标记 position 为 positive（在 known_modifications 中）或 negative
   - 用 `mod_ratio`、`stoichiometry`、`1-pvalue` 等列作为 score
   - 计算 AUPRC 和 AUROC（sklearn.metrics）
2. 输出一个汇总表：stoichiometry x scoring_mode x metric
3. 画一个 AUPRC vs stoichiometry 折线图（new vs legacy）

**注意**：position 匹配时 contig + position 直接对齐，known_modifications.tsv 和 pipeline 输出都是同一坐标系（POSITION_OFFSET=0）。

## Task 3: 运行 benchmark 并收集结果

> 依赖：Task 1 + Task 2 完成

执行 run_benchmark.sh，等全部跑完后执行 evaluate_benchmark.py。预期结果：

- 高 stoichiometry（50-100%）：new ≈ legacy（AUPRC 不退化）
- 低 stoichiometry（10-30%）：new >> legacy（AUPRC 从 ~1% 提升到 ~10%+）
- 如果低 stoich 没有改善，需要回查 V2 always-global alternative 是否生效（global params 可能被 degenerate 检查拒绝了）

## Task 4: 根据 benchmark 结果调整参数

> 依赖：Task 3 完成

根据 Task 3 的结果决定下一步：

- **A) AUPRC 改善显著且无退化** → 完成，清理代码
- **B) 高 stoich 退化** → 检查 degenerate global params 阈值（`sigma1 > 10*sigma0`, `sep > 50`）是否过松
- **C) 低 stoich 无改善** → global params 可能被拒绝：
  - 打印 `global_mu1`/`global_sigma1` 诊断
  - 考虑恢复 kNN contig-level rescue（但需要解决跨平台数值稳定性）
  - 考虑重新引入 V2 gate threshold 降低（sep 0.8→0.3）但加 legacy guard
- **D) stoichiometry 字段比 mod_ratio 更好** → 考虑把 stoichiometry 作为默认排序列

## Task 5: Squash commits 并清理 PR

> 依赖：Task 4 完成

当前 dev 上有多个 fix commit（threshold revert、kNN rescue removal 等）。最终确认效果后：

1. 考虑 squash 成 1-2 个干净的 commit
2. 更新 CLAUDE.md 如有架构变化
3. 确保 `--legacy-scoring` 文档清晰
4. 确保 CI 绿色

---

## 关键文件

| 文件 | 说明 |
|------|------|
| `baleen/eventalign/_hierarchical.py` | V2 always-global alternative + degenerate rejection |
| `baleen/eventalign/_aggregation.py` | 新增 `stoichiometry` 字段 |
| `baleen/eventalign/_pipeline.py` | `legacy_scoring` 参数传递 + `_GpuBudget` pickle 修复 |
| `baleen/cli.py` | `--legacy-scoring` CLI flag |
| `testdata/known_modifications.tsv` | 36 个 E. coli rRNA 已知修饰位点 ground truth |
| `testdata/prepare_bams.sh` | FASTQ → BAM 比对脚本（需要 minimap2 + samtools） |
