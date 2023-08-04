echo "Moving Jazz to $1"
python SupportingEndpoints/Test_Retrainer.py
mkdir -p RetrainingTests/$1
python SupportingEndpoints/Retrainer_ProcessDataset.py
python SupportingEndpoints/ErrorToMetric.py
mv Results RetrainingTests/$1/
mv *.dat   RetrainingTests/$1/ 
mv *.json  RetrainingTests/$1/ 
mv *.dat.png RetrainingTests/$1/
