cd /home/oovcharenko-ldap/projects/new_sempipes/sempipes

nohup poetry run python experiments/sigmod/rutgers/execute_sempipes_lightweight.py \
  > experiments/sigmod/rutgers/lightweight.log 2>&1 &
echo $! > experiments/sigmod/rutgers/lightweight.pid

nohup poetry run python experiments/sigmod/rutgers/execute_sempipes_lightweight_optimized.py \
  > experiments/sigmod/rutgers/lightweight_optimized.log 2>&1 &
echo $! > experiments/sigmod/rutgers/lightweight_optimized.pid

nohup poetry run python experiments/sigmod/rutgers/execute_sempipes_medium_optimized.py \
  > experiments/sigmod/rutgers/medium_optimized_3.1_pro_epsilon.log 2>&1 &
echo $! > experiments/sigmod/rutgers/medium_optimized.pid