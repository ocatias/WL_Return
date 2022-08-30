# Runs all benchmarks (except for CWN as that requires a different environment)
python Exp/run_experiment.py --repeats 10 --max_time 0.25 -grid Configs/Bench_GIN+SBE.yaml -dataset ZINC
python Exp/run_experiment.py --repeats 10 --max_time 0.25 -grid Configs/Bench_ESAN.yaml -dataset ZINC
python Exp/run_experiment.py --repeats 10 --max_time 0.25 -grid Configs/Bench_GIN+CRE.yaml -dataset ZINC
python Exp/run_experiment.py --repeats 10 --max_time 0.25 -grid Configs/Bench_GIN.yaml -dataset ZINC
python Exp/run_experiment.py --repeats 10 --max_time 0.25 -grid Configs/Bench_MLP.yaml -dataset ZINC