import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='GAUSSPROOF reproducible MIA benchmark')
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run',help='Train CNNs, evaluate attacks and export plots')
    run.add_argument('--config',type=Path,required=True)
    run.add_argument('--data',type=Path,required=True)
    run.add_argument('--output',type=Path,required=True)
    report=sub.add_parser('report',help='Export aggregate results and figures for publication')
    report.add_argument('--run',type=Path,required=True)
    report.add_argument('--output',type=Path,required=True)
    reconstruct=sub.add_parser('reconstruct',help='Learn public gradient priors and evaluate gradient/pixel reconstruction')
    reconstruct.add_argument('--config',type=Path,required=True)
    reconstruct.add_argument('--data',type=Path,required=True)
    reconstruct.add_argument('--output',type=Path,required=True)
    sub.add_parser('smoke',help='Run deterministic unit and gradient tests, no dataset needed')
    args=parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
    if args.command=='smoke':
        import unittest
        tests=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'))
        if tests.countTestCases() == 0:
            raise SystemExit('No tests found. Run smoke from an editable source checkout.')
        result=unittest.TextTestRunner(verbosity=2).run(tests)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.command=='report':
        if (args.run/'noise_prediction.csv').exists():
            from .diffusion_report import export as export_report
        elif (args.run/'gradient_summary.csv').exists():
            from .reconstruction_report import export_report
        else:
            from .report import export_report
        export_report(args.run,args.output)
        print(f'Exported aggregate report: {args.output}')
        return
    if args.command=='reconstruct':
        from .reconstruction import run_reconstruction
        run_reconstruction(json.loads(args.config.read_text()),args.data,args.output)
        return
    from .runner import run as run_benchmark
    run_benchmark(json.loads(args.config.read_text()),args.data,args.output)


if __name__=='__main__':
    main()
