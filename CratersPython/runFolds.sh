

for i in {1..10}; do python train_craters.py --fold $i --region East --gpus 0,1,2,3,4,5,6,7; done 2>&1 | tee logEast
for i in {1..10}; do python train_craters.py --fold $i --region Center --gpus 0,1,2,3,4,5,6,7; done 2>&1 | tee logCenter
for i in {1..10}; do python train_craters.py --fold $i --region West --gpus 0,1,2,3,4,5,6,7; done 2>&1 | tee logWest

cat logEast | grep "Epoch\[99\] Val"
cat logEast | grep "Epoch\[99\] Val" | awk '{ print $5}' | awk  -F= '{ total += $2; count++ } END { print total/count }'
