## Loss Function Variants

### Wasserstein GAN (WGAN)

WGAN introduces the Earth‐Mover (Wasserstein‑1) distance in place of the JS divergence, yielding more meaningful loss metrics and mitigating vanishing gradients during training ([arXiv][1]).

### WGAN with Gradient Penalty (WGAN‑GP)

Instead of weight clipping, WGAN‑GP enforces the Lipschitz constraint by penalizing the gradient norm of the critic’s output w\.r.t its input, which stabilizes training across architectures and datasets ([arXiv][2]).

### Least Squares GAN (LSGAN)

LSGAN replaces the sigmoid cross‐entropy loss with a least‐squares objective for the discriminator, which smooths gradients and reduces vanishing‐gradient problems, corresponding to minimizing the Pearson χ² divergence ([arXiv][3], [openaccess.thecvf.com][4]).

## Regularization and Normalization

### Spectral Normalization

Spectral normalization constrains the Lipschitz constant of each layer by normalizing its weight matrices’ spectral norms, offering a lightweight yet effective way to stabilize discriminator training ([arXiv][5]).

### Gradient Penalty Methods

Beyond WGAN‑GP, methods like DRAGAN (not detailed here) and various local perturbation penalties further regularize the gradient landscape, reducing mode collapse and convergence issues.

## Architecture and Training Schemes

### Progressive Growing of GANs (ProGAN)

By starting with low‐resolution images and progressively adding layers to both generator and discriminator, ProGAN accelerates convergence and produces high‑resolution outputs (up to 1024²) with remarkable stability ([arXiv][6]).

### Self‐Attention GAN (SAGAN)

SAGAN incorporates self‐attention modules to model long‐range dependencies, enabling fine details to be generated using context from all spatial locations in feature maps ([arXiv][7]).

### BigGAN

BigGAN scales both model capacity and batch size dramatically, revealing instabilities unique to large‑scale training but ultimately achieving state‑of‑the‑art inception scores on ImageNet by carefully tuning normalization and orthogonality constraints ([arXiv][8]).

### StyleGAN & StyleGAN2

StyleGAN introduces a style‐based generator architecture that disentangles high‑level attributes and stochastic variation, offering intuitive control over generated images; StyleGAN2 refines this further to eliminate artifacts and improve sample fidelity ([arXiv][9]).

## Advanced Techniques and Variants

### Unrolled GAN

Unrolled GAN defines the generator’s loss with respect to several “unrolled” steps of discriminator optimization, providing a smoother gradient signal and reducing mode collapse ([arXiv][10]).

### PacGAN

PacGAN modifies the discriminator to make joint decisions on packs of samples rather than individual ones, which penalizes generators that collapse to a few modes and thus mitigates mode collapse ([arXiv][11]).

### Relativistic GANs

Relativistic GANs train the discriminator to predict the probability that real data is more realistic than fake data, which accelerates convergence and improves sample quality over traditional GAN losses ([arXiv][12]).

### Two Time‐Scale Update Rule (TTUR)

TTUR uses different learning rates for the generator and discriminator—typically a slower generator and faster discriminator—to guarantee convergence to a local Nash equilibrium under mild assumptions ([arXiv][13]).

---

By combining these methods—choosing an appropriate loss, adding regularization, carefully designing the architecture, and tuning the training protocol—practitioners can significantly improve GAN stability and sample quality. Depending on the application, a hybrid approach (e.g., StyleGAN2 with TTUR and spectral normalization) often yields the best results.

[1]: https://arxiv.org/abs/1701.07875 "[1701.07875] Wasserstein GAN - arXiv"
[2]: https://arxiv.org/abs/1704.00028 "[1704.00028] Improved Training of Wasserstein GANs - arXiv"
[3]: https://arxiv.org/abs/1611.04076 "Least Squares Generative Adversarial Networks"
[4]: https://openaccess.thecvf.com/content_ICCV_2017/papers/Mao_Least_Squares_Generative_ICCV_2017_paper.pdf "[PDF] Least Squares Generative Adversarial Networks - CVF Open Access"
[5]: https://arxiv.org/abs/1802.05957 "Spectral Normalization for Generative Adversarial Networks - arXiv"
[6]: https://arxiv.org/abs/1710.10196 "Progressive Growing of GANs for Improved Quality, Stability ... - arXiv"
[7]: https://arxiv.org/abs/1805.08318 "[1805.08318] Self-Attention Generative Adversarial Networks - arXiv"
[8]: https://arxiv.org/abs/1809.11096 "Large Scale GAN Training for High Fidelity Natural Image Synthesis"
[9]: https://arxiv.org/abs/1812.04948 "[1812.04948] A Style-Based Generator Architecture for ... - arXiv"
[10]: https://arxiv.org/abs/1611.02163 "[1611.02163] Unrolled Generative Adversarial Networks - arXiv"
[11]: https://arxiv.org/abs/1712.04086 "The power of two samples in generative adversarial networks - arXiv"
[12]: https://arxiv.org/abs/1807.00734 "The relativistic discriminator: a key element missing from standard GAN"
[13]: https://arxiv.org/abs/1706.08500 "GANs Trained by a Two Time-Scale Update Rule Converge ... - arXiv"
