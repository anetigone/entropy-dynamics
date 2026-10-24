# 检索与来源记录

日期：2026-10-24（Asia/Shanghai）。范围：由 B1 的决定性近邻出发，标题检索、熵动态、早停与风险边界横向检索；不是整个 TTA 领域的系统综述。A-notes 与 frequency_dynamics/README 仅浏览，未修改。

## API

- 环境中 `OPENALEX_API_KEY` 存在，6 次 OpenAlex /works 查询均 HTTP 200。只报告存在性，不打印 key 或含 key 的请求 URL。请求式、返回文章 ID、作者、日期、DOI、开放地址、参考文献 ID、查询时引用计数均保存在 `search_log.json`；每组最多 8 条，按默认相关度，不按引用数筛选。
- Semantic Scholar 无 key，单次 /graph/v1/paper/search 返回 HTTP 429，未循环重试。该来源没有取得有效条目。
- OpenAlex broad search 噪声较大，如 early-stopping 查询混入通用回归结果。API 返回不是已读清单，也不是精确标题匹配结论。引用计数是查询快照，不作为质量或新颖性指标。
- `search.py` 可复跑；会再次调用 API，仅使用已有环境变量，不持久化凭据。后续引用图遍历尚未做。

## 网页检索式

实际使用包括：精确 WACV 标题；`test-time adaptation entropy collapse theory stopping time`；精确 Identifiable 标题；`test-time adaptation entropy hitting time`；`test-time adaptation entropy stopping theory 2026`；WACV 标题加 supplementary/supplemental；精确 `Layerwise Early Stopping for Test Time Adaptation`。官方 arXiv abs 页面用于版本核验。GALA 的改名用同 ID 去重。

## 阅读和抓取边界

浏览器成功读取 arXiv HTML：2606.26457v1、2605.28057v2、2402.09004v2、2604.17542v1、2404.03784v2、2311.18193v4；另从原始链接与 abs 页面核 Identifiable 信息。读取深度详见 `paper/RELATED_WORK.md`，不能将页面打开视为完整证明核验。

本地 requests 缓存成功：2606.26457v1、2604.17542v1、2609.11235v1、2311.18193v4 的 HTML 纯文本。2605.28057v2、2402.09004v2 本地请求 network_error，但浏览工具成功读取。WACV 官方 PDF 直接打开 403；搜索引擎返回官方正文 §3.1 与补充 Theoretical Analysis 的可读摘录，足以识别固定漂移假设，尚不能替代完整附件审读。未下载 PDF 批量库。

## 当前验收

问题、算法、比较器和最小非平凡现象已具体化；初稿包含标准基线推导和可复现 pilot。主定理、有效停止算法、视觉实验、新颖性均未完成。将“实际递推的风险安全时域”保留为研究假说，而非确认开放题。
