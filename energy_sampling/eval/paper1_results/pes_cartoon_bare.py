"""A text-free PES cartoon from figures.pes_cartoon with another seed: V(x) and its equilibrium (Boltzmann) P(x), or V(x)
alone with --no_p.
usage: python -m energy_sampling.eval.paper1_results.pes_cartoon_bare <out.png> [seed] [line width] [ruggedness]
    [barrier softness] [--no_p] [--p_height=F] (the tallest P(x) peak as a fraction of V's range, default 0.65)"""
import sys

from energy_sampling.eval.paper1_results.figures import pes_cartoon

if __name__ == '__main__':
    no_p = '--no_p' in sys.argv
    p_height = float(next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--p_height=')), 0.65))
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    out = args[0]
    seed = int(args[1]) if len(args) > 1 else 2026
    lw = float(args[2]) if len(args) > 2 else 6.0
    rug = float(args[3]) if len(args) > 3 else 1.0
    soft = float(args[4]) if len(args) > 4 else 0.0
    fig = pes_cartoon(seed=seed, distributions=() if no_p else ('equilibrium',), line_width=lw, bare=True,
                      ruggedness=rug, barrier_softness=soft, p_height=p_height)
    fig.write_image(out, width=900, height=540, scale=2)   # the paper cartoon's export size
    print('wrote', out)
