
CNNPROJECT.
BY BOUHADDA MOHAMED.



this project aims to evaluate the performance of cpu and gpu in running cnn models.
it create tensors and export model parameteres so then it give them to cuda files to run on gpu abd measure pergormanxe and also use them and run on cpp files on cpu and at last generate comparison.

python folder contain 4 python files 3 eich contains cnnonly model another poolingonly midel and a fully conected only model and at last a pipeline file that use the csv parametres genrayed by those files in export folder and create the pipeline.

cuda folder wich contains cu files that run the 4 python tensors on gpu and measure performances in order ( pipeline last).

cpuversions folder contains the cpu versions of the cu files that run the models and pipeline on cpu .

mnist and data files for mnist data byte files and normal files.


runa_ll.py: a  script that will automaticly run this project in order ( python files first then gpu abd cpu versions then the comparison and visualisation).
