#/bin/bash

args=("$@")
#echo $# arguments passed
if ! ((${#args[@]}==3)); then
  echo -e "Error: Need 3 arguments -- suggested usage\nsubmitTrail <startTrail> <endTrail> <queue>\n"
  exit 64
fi

start=${args[0]}
end=${args[1]}
queue=${args[2]}

cd Trail200/candidate

i=$start
while [ $i -le $end ]
do
  echo submitting trail $i
  cd $i
  depth=1
  while [ $depth -le 30 ]
  do
    echo Depth$depth
    cd Depth$depth
    shopt -s nullglob
    for filename in *.lmpdat
    do
      num=${filename%.lmpdat}
      num=${num#linker}
      submit_lammps_linker_deform_remote $num $queue
    done
    cd ../
    ((depth++))
  done
  cd ../
  ((i++))
done
