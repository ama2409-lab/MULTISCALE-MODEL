import argparse
import sys
# Ensure FINAL SET is importable
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--L', type=float, required=True)
    p.add_argument('--seeds', required=True, help='Comma-separated seeds')
    p.add_argument('--density', type=float, default=1e13)
    p.add_argument('--out', default='FINAL SET/sweep_rev_const_density_results.csv')
    p.add_argument('--plot', default='FINAL SET/sweep_rev_const_density_plot.png')
    p.add_argument('--rate', type=float, default=1e-12)
    p.add_argument('--thickness-fraction', type=float, default=0.05)
    args = p.parse_args()
    seed_list = [int(s) for s in args.seeds.split(',') if s.strip()]
    # Insert path so we can import sweep_rev_sizes
    sys.path.insert(0, r'.\FINAL SET')
    import sweep_rev_sizes as srs
    srs.process_L_seeds(args.L, seed_list, density=args.density, out=args.out, plot=args.plot, rate=args.rate, thickness_fraction=args.thickness_fraction)
