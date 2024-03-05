import os 
import pandas as pd

def table2fasta(table, file_out):
    file = open(file_out, 'w')
    for index, row in table.iterrows():
        file.write('>{0}\n'.format(row['id']))
        file.write('{0}\n'.format(row['seq']))
    file.close()
    print('Write finished')
    
def getblast(train, test):
    
    table2fasta(train, '/tmp/train.fasta')
    table2fasta(test, '/tmp/test.fasta')
    
    cmd1 = r'diamond makedb --in /tmp/train.fasta -d /tmp/train.dmnd --quiet'
    cmd2 = r'diamond blastp -d /tmp/train.dmnd  -q  /tmp/test.fasta -o /tmp/test_fasta_results.tsv -b5 -c1 -k 1 -e 1e-5 --quiet'
    cmd3 = r'rm -rf /tmp/*.fasta /tmp/*.dmnd /tmp/*.tsv'
    print(cmd1)
    os.system(cmd1)
    print(cmd2)
    os.system(cmd2)
    res_data = pd.read_csv('/tmp/test_fasta_results.tsv', sep='\t', names=['id', 'sseqid', 'pident', 'length','mismatch','gapopen','qstart','qend','sstart','send','evalue','bitscore'])
    os.system(cmd3)
    return res_data