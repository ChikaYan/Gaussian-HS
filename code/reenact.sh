


CONFS=()
CONFS+=(configs/ghs.conf)



CONF_RES=()
CONF_RES+=(confs/reenact/reenact_002.conf)
CONF_RES+=(confs/reenact/reenact_003.conf)



SUBJECTS=()
SUBJECTS+=("001")
SUBJECTS+=("002")
SUBJECTS+=("003")

for SUBJECT in "${SUBJECTS[@]}" ; do
    for CONF in "${CONFS[@]}" ; do
        for CONF_RE in "${CONF_RES[@]}" ; do

        echo $SUBJECT
        echo $CONF
        echo $CONF_RE
        
        python scripts/exp_runner.py --conf $CONF --subject $SUBJECT --is_reenact --conf_reenact $CONF_RE

        # reenact with network distillation
        python scripts/exp_runner.py --conf $CONF --subject $SUBJECT --is_reenact --conf_reenact $CONF_RE --run_fast_test

        done
    done
done









