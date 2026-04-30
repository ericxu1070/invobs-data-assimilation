import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

sigmas = [0.1, 0.5, 1.0]
colors = ['steelblue', 'darkorange', 'crimson']

x = np.linspace(-3.5, 3.5, 1000)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for ax, sigma, color in zip(axes, sigmas, colors):
    y = norm.pdf(x, loc=0, scale=sigma)
    ax.plot(x, y, color=color, lw=2.5)
    ax.fill_between(x, y, alpha=0.15, color=color)

    # ±1σ and ±2σ shading
    for n_sig, alpha in [(1, 0.25), (2, 0.12)]:
        mask = np.abs(x) <= n_sig * sigma
        ax.fill_between(x[mask], y[mask], alpha=alpha, color=color)

    # vertical lines at ±σ
    ax.axvline( sigma, color=color, lw=1, ls='--', alpha=0.7)
    ax.axvline(-sigma, color=color, lw=1, ls='--', alpha=0.7)
    ax.axvline(0, color='gray', lw=0.8, ls=':')

    ax.set_title(f'$\\sigma_{{obs}}$ = {sigma}', fontsize=13)
    ax.set_xlabel('noise value $\\epsilon$', fontsize=11)
    ax.set_xlim(-3.5, 3.5)
    ax.grid(alpha=0.3)
    ax.text(0.97, 0.95, f'68% within ±{sigma}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color=color)

axes[0].set_ylabel('probability density', fontsize=11)
fig.suptitle('Observation noise distributions  $\\epsilon \\sim \\mathcal{N}(0,\\, \\sigma_{{obs}}^2)$',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('noise_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved noise_distributions.png')
