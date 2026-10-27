"""A text-free PES cartoon: V(x) and its equilibrium (Boltzmann) P(x), from figures.pes_cartoon with another seed.
usage: python -m energy_sampling.eval.paper1_results.pes_cartoon_bare <out.png> [seed] [line width]"""
import sys

from energy_sampling.eval.paper1_results.figures import pes_cartoon

if __name__ == '__main__':
    out = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    lw = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
    fig = pes_cartoon(seed=seed, distributions=('equilibrium',), line_width=lw, bare=True)
    fig.write_image(out, width=900, height=540, scale=2)   # the paper cartoon's export size
    print('wrote', out)
