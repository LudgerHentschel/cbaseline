# Papers and citation

CBaseline is supported by two complementary technical papers by Ludger Hentschel:

1. [**Canonical Integrated Gradients: Expectations over Neutral Prediction Baselines**](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26h.pdf) (2026a).
   Develops the neutral-manifold reference distribution, expected Integrated
   Gradients, kernel localization, exponential calibration, and feasibility.
   See Sections 2–3 for the methodological foundation.
2. [**A Canonical Background Distribution for Shapley Attribution**](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26i.pdf) (2026b).
   Develops the Shapley interpretation and deterministic equal-weight
   construction. See Section 2 and Appendix A for the background methodology.

The guide brings the papers' reference-distribution viewpoint into practical
use. The papers provide the full arguments and empirical examples; the API
reference describes the implementation shipped with this version.

## Cite CBaseline

Please cite both papers when using the combined methodology, or the paper
corresponding to the construction and attribution method used in your work.
Also record the CBaseline version and background settings for reproducibility.

```bibtex
@misc{hentschel2026a,
  author = {Hentschel, Ludger},
  title = {Canonical Integrated Gradients: Expectations over Neutral Prediction Baselines},
  year = {2026},
  url = {https://www.ludgerhentschel.com/PDFs/Hentschel%20'26h.pdf},
}

@misc{hentschel2026b,
  author = {Hentschel, Ludger},
  title = {A Canonical Background Distribution for Shapley Attribution},
  year = {2026},
  url = {https://www.ludgerhentschel.com/PDFs/Hentschel%20'26i.pdf},
}
```

## Related references


- Hentschel, Ludger, 2026a, ["Canonical Integrated Gradients: Expectations over neutral prediction baselines."](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26h.pdf) *www.ludgerhentschel.com/Research.html*

- Hentschel, Ludger, 2026b, ["A canonical background distribution for Shapley attribution."](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26i.pdf) *www.ludgerhentschel.com/Research.html*

- Izzo, Cosimo, Aldo Lipani, Ramin Okhrati, and Francesca Medda, 2021, "A baseline for Shapley values in MLPs: From missingness to neutrality." *Proceedings of the 29th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN).*

- Lundberg, Scott M., and Su-In Lee, 2017, "A unified approach to interpreting model predictions." *Proceedings of the 31st International Conference on Neural Information Processing Systems.*

- Merrick, Luke, and Ankur Taly, 2020, "The explanation game: Explaining machine learning models using Shapley values." in *Machine Learning and Knowledge Extraction* (Holzinger, Andreas, Peter Kieseberg, A Min Tjoa, and Edgar Weippl, eds.)

- Sundararajan, Mukund, Ankur Taly, and Qiqi Yan, 2017, "Axiomatic attribution for deep networks." *Proceedings of the 34th International Conference on Machine Learning.*


## License

CBaseline is distributed under the [BSD 3-Clause License](https://github.com/LudgerHentschel/cbaseline/blob/main/LICENSE).
