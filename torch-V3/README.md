python -m cProfile -o torch-V2-30.prof torch-V2.py
python -m cProfile -o torch-V2-50.prof torch-V2.py
python -m cProfile -o torch-V2-40.prof torch-V2.py


python -m cProfile -o jax-V2-40.prof jax-V2.py

gprof2dot -f pstats --show-sample --node-label=total-time --node-label=self-time --node-label=self-time-percentage --node-label=total-time-percentage torch-V2-20.prof | dot -Tpng -o  torch-V2-20.png

gprof2dot -f pstats --show-sample --node-label=total-time --node-label=self-time --node-label=self-time-percentage --node-label=total-time-percentage torch-V2-20.prof | dot -Tpng -o  torch-V2-20.png

gprof2dot -f pstats --show-sample --node-label=total-time --node-label=self-time --node-label=self-time-percentage --node-label=total-time-percentage torch-V2-40.prof | dot -Tpng -o  torch-V2-40.png

gprof2dot -f pstats --show-sample --node-label=total-time --node-label=self-time --node-label=self-time-percentage --node-label=total-time-percentage scipy-V2.prof | dot -Tpng -o  scipy-V2-1.png