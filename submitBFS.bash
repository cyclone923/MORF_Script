#/bin/bash

args=("$@")
#echo $# arguments passed
if ! ((${#args[@]}==2)); then
  echo -e "Error: Need 3 arguments -- suggested usage\nsubmitBFS <depth> <queue>\n"
  exit 64
fi

depth=${args[0]}
queue=${args[1]}

cd ../testBFS/finalNode/depth$depth
for dir in linker*
do
  cd $dir
  num=${dir#linker}
  submit_lammps_linker_deform_remote $num $queue
  cd ../
done


