# i=0
# while [ $i -lt 500 ]
# do
#   submit_python3 moment-features_1022.py short 1 $i $i $((i+1))
#   i=$((i+1))
#   sleep 0.1
# done



# submit_python3 sideBackboneRatio.py short 2 n.d4.0 302 400
# i=400
# j=1
# while [ $i -lt 6300 ]
# do
#  submit_python3 sideBackboneRatio.py short 2 n.d4.$j $i $((i+100))
#  i=$((i+100))
#  ((j++))
# done
# submit_python3 sideBackboneRatio.py short 2 n.d4.$j 6300 6395

submit_python3 sideBackboneRatio.py short 2 n.d5.0 6395 7000
i=7000
j=1
while [ $i -lt 171000 ]
do
 submit_python3 sideBackboneRatio.py short 2 n.d5.$j $i $((i+2000))
 i=$((i+2000))
 ((j++))
done
submit_python3 sideBackboneRatio.py short 2 n.d5.$j 171000 171391



# i=110000
# j=2
# while [ $i -lt 170000 ]
# do
#  submit_python3 moment-features_1022.py short 2 $j $i $((i+2000))
#  i=$((i+2000))
#  ((j++))
#  sleep 0.1
# done

# submit_python3 moment-features_1022.py short 2 82 170000 171391
