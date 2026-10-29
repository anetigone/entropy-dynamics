# LaTeX 稿件

正文入口为 [main.tex](main.tex)，完整证明与实验细节入口为 [supplement.tex](supplement.tex)。使用仓库现有 `author-kit-CVPR2027-v1-latex` 样式，保留模板原件；工作稿暂不限制篇幅，不代表已完成投稿。

## 编译

从仓库根目录运行，需 TeX Live、latexmk 与 BibTeX：

```powershell
Set-Location paper/latex
$env:TEXINPUTS = '.;../../author-kit-CVPR2027-v1-latex/;'
$env:BSTINPUTS = '.;../../author-kit-CVPR2027-v1-latex/;'
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build supplement.tex
```

输出为 [正文 PDF](build/main.pdf)与[完整补充 PDF](build/supplement.pdf)。正文由 `sections/` 中的摘要、引言、过程与共同风险、统一退出证明、统计决策与双向归约、未知方向证明、证书与双层实验区间、讨论组成；`references.bib`维护其参考文献。

## 理论与来源

[Markdown 主文](../DRAFT.md)与[稿件导航](../README.md)提供连续阅读顺序。主文定理1–4及正文展开证明分别对应 L.2、L.3–L.7、L.9–L.10和 N.3；完整证明保留原标签。

补充材料依次包含 `supplement/local_prediction.tex`、`gaussian_decision.tex`、`unknown_direction.tex`、`gaussian_results.tex`、`confirmation.tex`，再收录一般标量退出、独立校准、风险接口、原高斯模型及证明、既有实验。原始 Markdown 与分析文件通过稿件导航定位，结果仍在原处。

`sections/`和 `supplement/`是当前整合后的 LaTeX 稿件，不通过源码指纹或登记清单维护。修改定理或结果时同步相应 Markdown 与 LaTeX，并重新编译。高斯数值证书、有限网格统计区间与渐近风险转移的范围须分别说明。

研究状态和下一步只见 [STATUS.md](../../STATUS.md)。

L.9–L.10 的 Markdown 源为 [gaussian_decision.md](../gaussian_decision.md)，N.1–N.4 的源为 [unknown_direction.md](../unknown_direction.md)，文献对照为 [RELATED_WORK.md](../RELATED_WORK.md)。这些文件已纳入 paper/，不再从原始包读取。
