# CVPR LaTeX 稿件

正文入口为 [main.tex](main.tex)，补充材料入口为 [supplement.tex](supplement.tex)。使用根目录 `author-kit-CVPR2027-v1-latex` 提供的样式和参考文献格式，保留模板原件。

从仓库根目录运行（需安装 TeX Live 与 BibTeX）：

```powershell
$env:TEXINPUTS = '.;../../author-kit-CVPR2027-v1-latex/;'
$env:BSTINPUTS = '.;../../author-kit-CVPR2027-v1-latex/;'
Set-Location paper/latex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build supplement.tex
```

输出为 `build/main.pdf` 与 `build/supplement.pdf`。正文由 `sections/` 中各部分组成。补充材料保留现有 Markdown 数学稿件的完整理论陈述、证明和详细实验；转换后的正文在 `supplement/`。理论原始编号仍按原稿解释，不与正文的新编号混用。

这是供讨论的英文初稿，使用 review 样式，不代表投稿完成。主张边界、当前缺口和后续安排统一见 [STATUS.md](../../STATUS.md)；逐项实验分析仍在原结果目录。
