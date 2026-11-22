#!/bin/zsh
# One locked resident session per schedule: `session.sh none|2|4`. A cold warm-up request binds the transformer,
# then the four fixed prompts at seed 7, warm; vm.loadavg before and after each request.
zmodload zsh/system
D=${0:A:h}; M=/Users/agents/models/MiniMax-H3-Image; S=$1
B=${TF_BIN:-$D/../../../zig-out/bin}/tf-h3-image-server
flags=(); [[ $S != none ]] && flags=(--draft-steps $S)
zsystem flock -f lk /tmp/tensorfold-model.lock || exit 3
coproc $B $M/text_encoder $M $flags 2>&1
while read -r -p line; do echo "$line" >> $D/$S.out; [[ $line == *ready* ]] && break; done
for p in warmup text faces texture architecture; do
  echo "$S $p load-before $(sysctl -n vm.loadavg)" >> $D/loads.txt
  print -p "2048x2048 7 $D/$S-$p $D/${p/warmup/text}.txt"
  while read -r -p line; do echo "$line" >> $D/$S.out; [[ $line == "$D/$S-$p:"*total* || $line == *rror* ]] && break; done
  echo "$S $p load-after $(sysctl -n vm.loadavg)" >> $D/loads.txt
done
coproc exit
wait
zsystem flock -u $lk
