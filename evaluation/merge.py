import os 

ncbis = "10665 11698 9796 11706 573 11686 83332 11676 186497 273057 4081 510516 287 5116 36329 273063 11678 694009 93062 264203 2697049 9031 9606 83333 1280 562".split()
data_path = "models/output/ProteinNet"
target_path = os.path.join(data_path, 'pretrain')
os.system(f"mkdir -p {target_path}")
os.system(f"mkdir -p {target_path}/pred_pdbs")
os.system(f"mkdir -p {target_path}/tgt_pdbs")

for idx, ncbi in enumerate(ncbis):
    if idx == 0:
        os.system(f'cp {data_path}/{ncbi}/log_likelihood.txt {target_path}')
        os.system(f'cp {data_path}/{ncbi}/pdb.txt {target_path}')
        os.system(f'cp {data_path}/{ncbi}/protein.txt {target_path}')
        os.system(f'cp {data_path}/{ncbi}/src.seq.txt {target_path}')
    else:
        os.system(f'cat {target_path}/log_likelihood.txt {data_path}/{ncbi}/log_likelihood.txt > {target_path}/log_likelihood.new.txt')
        os.system(f'mv {target_path}/log_likelihood.new.txt {target_path}/log_likelihood.txt')
        os.system(f'cat {target_path}/pdb.txt {data_path}/{ncbi}/pdb.txt > {target_path}/pdb.new.txt')
        os.system(f'mv {target_path}/pdb.new.txt {target_path}/pdb.txt')
        os.system(f'cat {target_path}/protein.txt {data_path}/{ncbi}/protein.txt > {target_path}/protein.new.txt')
        os.system(f'mv {target_path}/protein.new.txt {target_path}/protein.txt')
        os.system(f'cat {target_path}/src.seq.txt {data_path}/{ncbi}/src.seq.txt > {target_path}/src.seq.new.txt')
        os.system(f'mv {target_path}/src.seq.new.txt {target_path}/src.seq.txt')
        
    
    pdbs = [line.strip() for line in open(os.path.join(data_path, ncbi, "pdb.txt")).readlines()]
    for pdb in pdbs:
        os.system(f'cp {data_path}/{ncbi}/pred_pdbs/{pdb}.pdb {target_path}/pred_pdbs')
        os.system(f'cp {data_path}/{ncbi}/tgt_pdbs/{pdb}.pdb {target_path}/tgt_pdbs')
        