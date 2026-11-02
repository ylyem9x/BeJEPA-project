# BeJEPA project website

[Project Page](https://ylyem9x.github.io/BeJEPA-project/) | [Paper](https://arxiv.org/abs/2610.23127) | [Code](https://github.com/ylyem9x/BeJEPA-project)

A black-and-white academic project page for **BeJEPA: A Goal-Conditioned Humanoid Behavior World Model via Unsupervised RL**, with detailed tracking comparisons and experimental figures.

The `gh-page` branch contains the website and its media. The `main` branch contains the BeJEPA research code and bundled motion data. The website links to the public code repository and arXiv paper, and includes the paper’s BibTeX citation.

## Editing

- `index.html`: lab identity, title, static author fallback, abstract, figures, a four-equation method explanation, and three results sections with benchmark tables and figures.
- `styles.css`: responsive monochrome layout.
- `site.config.js`: author links, affiliations, corresponding-author flags, the paper and code URLs, and citation.
- `script.js`: author rendering, paper and code links, and video visibility handling.
- `author.md`: author order and institution-order requirements supplied by the project owner.
- `assets/`: the supplied RL² Lab logo, full project video, WebM fallback, captions, four paper figures, posters, favicon, and a locally hosted font.

Author order and affiliation assignments follow `author.md`; Yue Gao is marked as corresponding author. Institution order is RL² Lab at Shanghai Jiao Tong University, SII, Zhejiang University, Tongji University, Fudan University, and Lenovo last. The owner confirmed that `tju` means Tongji University, and requested the RL² Lab attribution within the SJTU affiliation. When updating authors, keep the static fallback in `index.html` consistent with `site.config.js`.

Keep `arxivUrl`, `codeUrl`, and `citation` in `site.config.js` consistent with the static links and BibTeX in `index.html`, so they remain available when JavaScript is disabled.

GitHub Pages serves the root of `gh-page`, preserving https://ylyem9x.github.io/BeJEPA-project/. There is no build step and no external JavaScript or font CDN dependency. A local static HTTP server can preview the page; byte-range support is needed to seek through videos.

## Media and results

The complete video and descriptive captions come from the supplied supplementary materials. The teaser, architecture, MPC trade-off plot, and real-robot experiment montage are rendered from the supplied paper figures. The lab logo is the image supplied by the project owner. Original logo, figure, and video colors are preserved; the page interface is monochrome.

Key results begin directly with the full AMASS and LAFAN tables, reproducing the paper's tracking table, including best and second-best values. Comparisons use selected, non-interactive 20-second G1 tracking experiments in MuJoCo at 50 Hz, retaining trajectories with large spatial deviations. The MPC plot illustrates execution priorities and temporal alignment; the hardware montage shows disturbance recovery, ground recovery, and goal switching. No manuscript or research code is distributed by this website.

The method explanation follows the paper's goal-reward, Dynamic prediction, Successor-feature, and composed-action-value equations. Equations are pre-rendered with KaTeX 0.19.0, with locally hosted styles and fonts; no browser rendering script or CDN is required. Each formula retains its LaTeX source in its accessible MathML annotation. The RL² favicon displays the emblem portion of the supplied lab banner through an SVG viewport; the original image is embedded unchanged.

The bundled Inter font retains its SIL Open Font License at `assets/fonts/OFL.txt`. KaTeX retains its MIT license at `assets/katex/LICENSE`; its CSS is configured to use the bundled WOFF2 fonts. Research media retain their original rights.

The academic layout references [Demo-JEPA](https://log2r.github.io/Demo-JEPA/), with expanded tables, explanatory text, and figures, following earlier references to [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) and [UMI on Legs](https://umi-on-legs.github.io/). No third-party template code or research assets were copied.

## Citation

```bibtex
@misc{yan2026bejepa,
  title         = {{BeJEPA}: A Goal-Conditioned Humanoid Behavior World Model via Unsupervised RL},
  author        = {Liyun Yan and Haoyang Li and Yang Zhang and Jialing Zhu and Zhanxiang Cao and Zihao Zhou and Keqi Zhu and Yizhi Chen and Huang Chuxuan and Shengcheng Fu and Zhong Jiangwei and Qiao Sun and Chen Zhang and Yue Gao},
  year          = {2026},
  eprint        = {2610.23127},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2610.23127},
  url           = {https://arxiv.org/abs/2610.23127}
}
```
