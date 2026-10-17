#!/bin/sh
# swap.sh BASE_KEY TIP_KEY BASE_TABLE TIP_TABLE: each build reads one table under its own key, both tables in turn.
# homes/with-base-table: the base's table under both keys; homes/with-tip-table: the tip's table under both keys.
set -eu
cd "$(dirname "$0")"
kb=$1 kt=$2 tb=$3 tt=$4
for h in base:$tb tip:$tt; do
    name=${h%%:*} table=${h#*:}
    d=homes/with-$name-table/.cache/tensorfold/window-costs
    rm -rf homes/with-$name-table && mkdir -p $d
    cp $table $d/$kb.bin
    cp $table $d/$kt.bin
done
set +e
for item in base-btable-1:base:ed44eefbaefdfc6b52c0740ac9aeda3bbc5a36ba:base tip-btable-1:tip:14c78e735cc9effb2dfb2c2dc53c54b93d06f481:base tip-btable-2:tip:14c78e735cc9effb2dfb2c2dc53c54b93d06f481:base base-btable-2:base:ed44eefbaefdfc6b52c0740ac9aeda3bbc5a36ba:base tip-ttable-1:tip:14c78e735cc9effb2dfb2c2dc53c54b93d06f481:tip base-ttable-1:base:ed44eefbaefdfc6b52c0740ac9aeda3bbc5a36ba:tip base-ttable-2:base:ed44eefbaefdfc6b52c0740ac9aeda3bbc5a36ba:tip tip-ttable-2:tip:14c78e735cc9effb2dfb2c2dc53c54b93d06f481:tip; do
    label=${item%%:*} rest=${item#*:} dir=${rest%%:*} rest=${rest#*:} commit=${rest%%:*} table=${rest#*:}
    flock -w 7200 <gpu.lock> timeout 20m python3 hold.py $label $PWD/../builds/$dir/tensorfold-native $commit $PWD/../builds/tools $PWD/homes/with-$table-table concurrent
    echo "$label exit=$?"
done
