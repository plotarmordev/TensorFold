#!/bin/sh
# Desk-only staging for card 242: v0.2.19's Inkling engine and Control, plus the tokenizer.
# SQ-20 already lives in strata-inkling-sq20/ and SQ-10 in strata-inkling-pack/ on TOP.
# Do not upload either pack again. The Desk runs this from a machine with network and top-menu access.
set -eu
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
dir=$work/strata-inkling-0219
mkdir "$dir"
release=https://github.com/Niko1221/Strata/releases/download/v0.2.19
packs=https://huggingface.co/strata-engine/Inkling-2-Mini-Strata/resolve/04996a1e2c45009b6a04f2af126cb6bdb2ba3587
for entry in 'inkling-0.1.65-linux-aarch64-cuda-cuda13 2a89c1ded0ceefccff7cf2130fd35597869d0113d72cb3fd27bc2a9006fa03cc' 'control-0.1.65-linux-aarch64 177269e2164221d43314bcb2df45aa33dc23b0902f544e2cc0af0775cf060de3'; do
    set -- $entry
    curl -sSfL -o "$work/strata-$1.tar.gz" "$release/strata-$1.tar.gz"
    echo "$2  $work/strata-$1.tar.gz" | sha256sum -c --quiet
    mkdir "$dir/${1%%-*}"
    tar -C "$dir/${1%%-*}" -xzf "$work/strata-$1.tar.gz"
done
curl -sSfL -o "$dir/inkling_tokenizer.py" https://raw.githubusercontent.com/Niko1221/Strata/v0.2.19/terminal/tools/inkling_tokenizer.py
echo '59ce0ae7bb7eeeb3797744eeab4d77098154c03625e819c8ba550e19f7ddadef  '"$dir/inkling_tokenizer.py" | sha256sum -c --quiet
curl -sSfL -o "$dir/tokenizer.json" "$packs/tokenizer.json"
echo '7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6  '"$dir/tokenizer.json" | sha256sum -c --quiet
tar -C "$work" -c strata-inkling-0219 | ssh top-menu 'put top strata-inkling-0219'
