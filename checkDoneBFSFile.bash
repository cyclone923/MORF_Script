#/bin/bash

# i=5
# if [ $((i%5)) -eq 0 ] || [ $((i%5)) -eq 1 ]
# then
#   echo short
# else
#   echo greaneylab
# fi

args=("$@")
#echo $# arguments passed
if ! ((${#args[@]}==2)); then
  echo -e "Error: Need 2 arguments -- suggested usage\nsubmitBFS <depth>,<fileName>\n"
  exit 64
fi

depth=${args[0]}
file=${args[1]}

cd BFS/finalNode/depth$depth
for dir in linker*
do
  num=${dir#linker}
  # echo $num
  cd $dir
  # pwd
  # if [ -e ./linker${num}_deformation/$file ]
  if [ -e ./linker${num}_deformation/linker${num}-$file ]
  then
    true
  else
    echo linker${num}-$file Miss
    # submit_lammps_linker_deform_remote $num greaneylab
  fi
  cd ../
  # exit 0
done
echo Done
