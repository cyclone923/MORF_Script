#/bin/bash

args=("$@")
#echo $# arguments passed
if ! ((${#args[@]}==2)); then
  echo -e "Error: Need 3 arguments -- suggested usage\nsubmitTrail <startTrail> <endTrail>\n"
  exit 64
fi

start=${args[0]}
end=${args[1]}

cd Trail200/candidate

i=$start
while [ $i -le $end ]
do
  echo check trail $i
  ifcheck=1
  cd $i
  depth=1
  while [ $depth -le 30 ]
  do
    cd Depth$depth
    shopt -s nullglob
    for filename in *deformation
    do
      if [ $ifcheck -eq 1 ]
      then
        cd $filename
        num=${filename%_deformation}
        num=${num#linker}
        if [ -e linker${num}-modified-features.txt ]
        then
          #echo Trail$i Depth$depth linker${num}-features.txt exist
	  true
        else
          echo Trail$i Depth$depth linker${num} Miss
	  #dir=$PWD
	  #cd ~/shared/CleanMORF/randomOutput
	  #submit_python3 moments-features_1011.py batch 2 $i $i $((i+1))
	  #cd $dir
	  ifcheck=0
        fi
        cd ../
      fi
    done
    cd ../
    ((depth++))
  done
  cd ../
  ((i++))
done
