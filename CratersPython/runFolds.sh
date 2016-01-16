

for i in {1..10}; do python train_craters2.py --fold $i --region East; done 2>&1 | tee logEast
for i in {1..10}; do python train_craters2.py --fold $i --region Center; done 2>&1 | tee logCenter
for i in {1..10}; do python train_craters2.py --fold $i --region West; done 2>&1 | tee logWest

cat logEast | grep "Epoch\[49\] Val"
cat logEast | grep "Epoch\[49\] Val" | awk '{ print $2}' | awk  -F= '{ total += $2; count++ } END { print total/count }'
