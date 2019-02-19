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
  cd $i
  depth=1
  while [ $depth -le 30 ]
  do
    cd Depth$depth
    shopt -s nullglob
    for filename in *deformation
    do       
      cd $filename
      num=${filename%_deformation}
      num=${num#linker}
      if [ -e DONE ]
      then
        if [ -e log.lammps ]
        then
          echo Depth$depth
          echo $filename Recompute
          cd ../
          submit_lammps_linker_deform_remote ${num} greaneylab
	        cd $filename
        else
          if [ ! -e linker${num}-ave-force.d ]
          then
            echo yes
            echo Depth$depth
            echo $filename Ave not found
            cd ../
            submit_lammps_linker_deform_remote ${num} greaneylab
	          cd $filename
          fi
        fi
        true
      else
        true
        echo Depth$depth
        echo $filename Not finished
        # cd ../
        # submit_lammps_linker_deform_remote ${num} greaneylab
	      # cd $filename
      fi
      cd ../
    done
    cd ../
    ((depth++))
  done
  cd ../
  ((i++))
done
