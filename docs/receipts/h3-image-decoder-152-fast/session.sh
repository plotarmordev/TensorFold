#!/bin/zsh
# one locked fast session per build: pilot (seed 42) then three unseen prompts; load before/after each request
zmodload zsh/system
D=/tmp/b152-run; M=/Users/agents/models/MiniMax-H3-Image
for b in "$@"; do
  zsystem flock -f lk /tmp/tensorfold-model.lock || exit 3
  coproc /tmp/b152-$b/bin/tf-h3-image-server $M/text_encoder $M --draft-steps 3 2>&1
  while read -r -p line; do echo "$line" >> $D/$b.out; [[ $line == *ready* ]] && break; done
  for req in "42 pilot" "31 v1" "32 v2" "33 v3"; do
    s=${req% *}; p=${req#* }
    echo "$b $p load-before $(sysctl -n vm.loadavg)" >> $D/loads.txt
    print -p "2048x2048 $s $D/$b-$p $D/$p.txt"
    while read -r -p line; do echo "$line" >> $D/$b.out; [[ $line == "$D/$b-$p:"*total* ]] && break; done
    echo "$b $p load-after $(sysctl -n vm.loadavg)" >> $D/loads.txt
  done
  coproc exit  # closes the server's stdin: it quits at EOF
  wait
  zsystem flock -u $lk
done
