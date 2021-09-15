set -e

n_gbs=$1
gb_to_b=1000000000

let n_bs=n_gbs*gb_to_b

plasma_store -m $n_bs -s /tmp/plasma