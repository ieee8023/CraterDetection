
#for i in {1..10}; do python train_craters2.py --fold $i; done 2>&1 | tee log

cat log | grep "Epoch\[49\] Val"
cat log | grep "Epoch\[49\] Val" | awk '{ print $2}' | awk  -F= '{ total += $2; count++ } END { print total/count }'
