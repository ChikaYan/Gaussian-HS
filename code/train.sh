

CONFS=()
CONFS+=(configs/ghs.conf)



SUBJECTS=()
SUBJECTS+=("001")
# SUBJECTS+=("002")
# SUBJECTS+=("003")


for CONF in "${CONFS[@]}" ; do
    for SUBJECT in "${SUBJECTS[@]}" ; do

        echo $SUBJECT

        python scripts/exp_runner.py --conf $CONF --subject $SUBJECT --quick_eval # add --quick_eval to use a subset of the test data for quick evaluation
        
        python scripts/exp_runner.py --conf $CONF --subject $SUBJECT --is_eval --quick_eval --run_fast_test --wandb_tags "quick_distill_eval_50"

        python scripts/exp_runner.py --conf $CONF --subject $SUBJECT --is_eval --wandb_tags "full_eval_25"

        python scripts/exp_runner.py --conf $CONF --subject $SUBJECT --is_eval --run_fast_test --wandb_tags "full_distill_eval_50"




    done
done


