#!/usr/bin/env bash
# Download raw public spectral datasets reachable from GitHub (raw.githubusercontent.com).
set -u
R=/home/claude/spectwin/data/raw
get() { # get <subdir> <url-without-branch-prefix> <branch1> [branch2]
  local sub=$1 path=$2; shift 2
  mkdir -p "$R/$sub"; local f="$R/$sub/$(basename "$path")"
  [ -s "$f" ] && { echo "have  $sub/$(basename "$path")"; return; }
  for br in "$@"; do
    url="${path/BRANCH/$br}"
    if curl -sfL -m 300 -o "$f" "$url"; then echo "ok    $sub/$(basename "$path") ($(du -h "$f" | cut -f1))"; return; fi
  done
  rm -f "$f"; echo "FAIL  $sub/$(basename "$path")"
}
C=https://raw.githubusercontent.com/cran
for x in ppls/cookie.rda pls/mayonnaise.RData chemometrics/NIR.rda chemometrics/cereal.rda caret/tecator.RData \
         rchemo/cassav.rda rchemo/forages.rda multiblock/emulsions.rda multiblock/potato.rda EMSC/fishoil.rda mdatools/carbs.RData; do
  pkg=${x%%/*}; file=${x#*/}; get cran_$pkg "$C/$pkg/BRANCH/data/$file" master main
done
J=https://raw.githubusercontent.com/mlesnoff/JchemoData.jl/BRANCH/data
for f in corn.jld2 caltransfer.jld2 challenge2018.jld2 challenge2021.jld2 sorghum.jld2 grapes.jld2 oregano_occ.jld2 \
         forages2.jld2 wheatkernels_cui.jld2 fermentation.jld2 cassav_X.csv cassav_Y.csv tecator.xlsx dat_2021.csv \
         datspir_X.csv datspir_Y.csv datspir_M.csv mango.mat machine.mat; do get jchemodata "$J/$f" main master; done
D=https://raw.githubusercontent.com/dario-passos/DeepLearning_for_VIS-NIR_Spectra/BRANCH/notebooks
for f in Deep-Tuttifrutti_I/data/NAnderson2020MendeleyMangoNIRData.csv DeepSpectra2019/corn.mat CuiFearn2018_CNN/NITSingleSeed.MAT \
         KanekoFunatsu2015_SVROpt/spectra_pears.csv KanekoFunatsu2015_SVROpt/brix_pears.csv KanekoFunatsu2015_SVROpt/lambda.txt \
         Bjerrum2017_CNN/nir_shootout_2002.mat; do get passos "$D/$f" master main; done
N=https://raw.githubusercontent.com/nevernervous78/nirpyresearch/BRANCH/data
for f in peach_spectra_brix.csv NIRplums_brix_firmness.csv plums.csv milk.csv milk-powder.csv Honey2021.csv \
         coffee_classification.csv NIR_avocado_spectra_nirpy.zip Raman_spectra_mixtures.csv; do get nirpy "$N/$f" master main; done
