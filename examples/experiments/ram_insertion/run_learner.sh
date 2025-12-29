export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=ram_insertion \
    --checkpoint_path=first_run \
    --demo_path=demo_data/ram_insertion_20_demos_2025-12-28_21-04-19.pkl\
    --learner \