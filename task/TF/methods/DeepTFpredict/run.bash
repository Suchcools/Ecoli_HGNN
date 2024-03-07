git clone https://bitbucket.org/kaistsystemsbiology/deeptfactor.git
cd deeptfactor
conda env create -f environment.yml
conda activate deeptfactor
python tf_running.py -i output/test.fasta -o output/ -g cuda:0