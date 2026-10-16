# BeJEPA project website

https://ylyem9x.github.io/BeJEPA-project/

A black-and-white academic project page for **BeJEPA: A Goal-Conditioned Humanoid Behavior World Model via Unsupervised RL**, with detailed tracking comparisons and experimental figures.

This repository contains only the website and the media used by the page. Research implementation, code archives, datasets, model weights, and manuscript PDFs are not included. Code is marked **Coming soon**, and the citation is intentionally empty until the arXiv release.

## Editing

- `index.html`: lab identity, title, static author fallback, abstract, figures, method, and key results with benchmark tables.
- `styles.css`: responsive monochrome layout.
- `site.config.js`: author links, affiliations, corresponding-author flags, and the future citation.
- `script.js`: author rendering and video visibility handling.
- `author.md`: author order and institution-order requirements supplied by the project owner.
- `assets/`: the supplied RL² Lab logo, full project video, WebM fallback, captions, four paper figures, posters, favicon, and a locally hosted font.

Author order and affiliation assignments follow `author.md`; Yue Gao is marked as corresponding author. Institution order is RL² Lab at Shanghai Jiao Tong University, SII, Zhejiang University, Tongji University, Fudan University, and Lenovo last. The owner confirmed that `tju` means Tongji University, and requested the RL² Lab attribution within the SJTU affiliation. When updating authors, keep the static fallback in `index.html` consistent with `site.config.js`.

Set `citation` in `site.config.js` to the final BibTeX only after the preprint is released. No placeholder citation is generated.

GitHub Pages serves the root of `main`. There is no build step and no external JavaScript or font CDN dependency. A local static HTTP server can preview the page; byte-range support is needed to seek through videos.

## Media and results

The complete video and descriptive captions come from the supplied supplementary materials. The teaser, architecture, MPC trade-off plot, and real-robot experiment montage are rendered from the supplied paper figures. The lab logo is the image supplied by the project owner. Original logo, figure, and video colors are preserved; the page interface is monochrome.

The headline results are AMASS global MPJPE of **219.5 mm** (BeJEPA-MPC Balanced), and mean signed tracking lag of **94.8 → 21.2 ms** (BeJEPA → BeJEPA-MPC Balanced). The full AMASS and LAFAN tables reproduce the paper's tracking table, including best and second-best values. Comparisons use selected, non-interactive 20-second G1 tracking experiments in MuJoCo at 50 Hz, retaining trajectories with large spatial deviations. The MPC plot illustrates execution priorities and temporal alignment; the hardware montage shows disturbance recovery, ground recovery, and goal switching. No manuscript or research code is distributed by this website.

The bundled Inter font retains its SIL Open Font License at `assets/fonts/OFL.txt`. Research media retain their original rights.

The academic layout references [Demo-JEPA](https://log2r.github.io/Demo-JEPA/), with expanded tables, explanatory text, and figures, following earlier references to [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) and [UMI on Legs](https://umi-on-legs.github.io/). No third-party template code or research assets were copied.
