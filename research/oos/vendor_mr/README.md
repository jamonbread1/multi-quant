# vendor_mr — 从 `market_regime -git.zip` 原样复制的最小依赖子集

来源: `market_regime -git.zip` → `market_regime/` (未修改任何文件内容)
- features.py, taxonomy.py, detectors/base.py, detectors/rules.py (v1.0), detectors/rules_v1_1.py (v1.1)

未复制 (与本研究无关, 且依赖 sklearn/hmmlearn): pipeline.py, hmm.py, gmm.py, hybrid.py, allocation.py 等。
`__init__.py` 为空, 由 `state_transition_v1_0.py` 以 stub 方式导入子模块。
