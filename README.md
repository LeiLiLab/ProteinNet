<h1>ProteinNet Is a Large Generative Model for Ligand-Based Functional Protein Sequence and Structure Co-Design</h1>

<h2>Model Architecture</h2>

This repository contains code, data and model weights.

The overall model architecture is shown below:

![image](./ProteinNet_overall.png)


<h2>Environment</h2>
The dependencies can be set up using the following commands:

```ruby
conda env create -f proteinnet.yml
conda activate proteinnet
bash setup.sh 
```

<h2>Download Data</h2>

We provide the pretraining, finetuning, and evaluation data at [ProteinNet_Data](https://drive.google.com/drive/folders/1rn5FolLZ2NlFmPDqnbUTJEYS7brCr-w9?usp=sharing) 
 and NCBI taxonomy category ID to index dict at [NCBI_ID_Mapping_Dict](https://drive.google.com/file/d/1qe3T3-1z9L8h-e27O5i3Gsdak4A1AZG0/view?usp=sharing)

Please download the dataset and put them in the data folder.

First if you want to pretrain your own model, please download the pretraining data:

```angular2html
mkdir data 
cd data 
wget https://drive.google.com/file/d/1ROcJTMfBIXlS1iUIqSE5Dtww1OC2GHYt/view?usp=sharing
```

Then if you want to finetune your own model, please download the finetuning data:

```angular2html
wget https://drive.google.com/file/d/1dGzW1D95G86HU02UytDmw9XHGbyQMlpm/view?usp=drive_link
wget https://drive.google.com/file/d/1N2Z7-YhSFiO6-Ef7ytr3y7wy_SxgMMUj/view?usp=sharing
wget https://drive.google.com/file/d/1AQDFDT0Ps3_SbKivAAvjyzCCuDmfosjl/view?usp=sharing
```

Then please download the NCBI taxonomy id mapping dict which is necessary for running the code:

```angular2html
wget https://drive.google.com/file/d/1qe3T3-1z9L8h-e27O5i3Gsdak4A1AZG0/view?usp=sharing
```

Then please download the evaluation data:

```angular2html
wget https://drive.google.com/file/d/1g5lI2jFKPe1m6U8eu4Onsw7mRkFAA4IT/view?usp=sharing
```

<h2>Download Model</h2>

We provide the pretrained and finetuned model checkpoints used in the paper at [Models](https://drive.google.com/drive/folders/12th0ZFG0N8YdKjUyaY0t9-YzIvELjncX?usp=sharing) 


Please download the checkpoints and put them in the models folder.

<h3>Download the pretrained model weights</h3>

```ruby
mkdir models
mkdir models/ProteinNet
cd models/ProteinNet
wget https://drive.google.com/file/d/1PZMKNDDTXZPofZX8Lu-QZ7OhYEwyHhWG/view?usp=sharing
```

<h3>Download the finetuned model weights</h3>

ChlR:

```ruby
mkdir models/rhea_18421_finetune
cd models/rhea_18421_finetune
wget https://drive.google.com/file/d/1DN1fbrf76brN6qvCCInRVWrP8F6-5xcA/view?usp=sharing
```

AadA:

```ruby
mkdir models/rhea_20245_finetune
cd models/rhea_20245_finetune
wget https://drive.google.com/file/d/1cJiqFgOgjeGkQ0SZX1Fyw-pIzeu9wh0A/view?usp=sharing
```

TPMT:

```ruby
mkdir models/rhea_Thiopurine_S_methyltransferas_finetune
cd models/rhea_Thiopurine_S_methyltransferas_finetune
wget https://drive.google.com/file/d/12WR0_TDlobEaFI7TYAZOn8adUz4PrBb0/view?usp=sharing
```


If you want to pretrain or finetune your own model, please follow the training guidance below. Otherwise, you can directly go to the Inference section.

<h2>Pretraining</h2>
If you want to pretrain a model with protein-ligand interaction constraint as introduced in our paper, please follow the script below. Our pretraining process involves three stages. First the model is pretrained only on the sequence prediction loss and structure reconstructure loss with 20% residues are masked and 80% are given:

```ruby
bash train_ProteinNet_mlm.sh
```

Then conditioned on the model pretrained in the first stage, the model continues to be trained on the sequence prediction loss and structure reconstructure loss with motifs are given:

```ruby
bash train_ProteinNet_motif.sh
```

Finally conditioned on the model pretrained in the second satge, the model continues to be trained on the full losses, including the sequence prediction loss, structure reconstructure loss and protein-ligand interaction prediction loss:

```ruby
bash train_ProteinNet_full.sh
```

<h2>Finetuning</h2>

To finetuning the model on a specific protein family, which are chloramphenicol acetyltransferase (ChlR), aminoglycoside adenylyltransferase (AadA), and thiopurine methyltransferase (TPMT) in our paper, please follow the guidance below:

Finetuning the pretrained model on ChlR:

```ruby
bash reah_ChlR_finetune.sh
```

Finetuning the pretrained model on AadA:

```ruby
bash reah_AadA_finetune.sh
```

Finetuning the pretrained model on TPMT:

```ruby
bash reah_TPMT_finetune.sh
```


<h2>Inference</h2>
To design proteins of the 10 largest enzymes in our test set using the pretrained model, please use the following scripts:

```ruby
bash generate_proteinnet_pretrain.sh
```

There are six items in the output directory:

1. protein.txt refers to the designed protein sequence
2. src.seq.txt refers to the reference protein sequences
3. pdb.txt refers to the target PDB ID and the corresponding chain
4. log_likelihood.txt refers to the log likelihood of the designed protein sequence
5. pred_pdbs refers to the directory of designed protein structures
6. tgt_pdbs refers to the directory of reference protein structures

To design the enzymes for the three finetuned enzymes, follow the guidance below:

ChlR:

```ruby
bash generate_ChlR.sh
```

AadA:

```ruby
bash generate_AadA.sh
```

TPMT:

```ruby
bash generate_TPMT.sh
```

<h2>Designing Your Own Protein</h2>

If you want to design your own protein, follow the pipeline below:

First, you'll need to prepare your own data, we provide the example of design beta-lactam antibiotics with the motif from PDB entry 3DWZ:

```ruby
cd example
python prepare_example_data.py --pdb_file_path 3DWZ.cif --protein_id 3DWZ --motif "6,8,9,14-18,23,30-44,47,49-52,54,60,64,66,70,75-78,80-87,94-118,125,126,129,130,132,134-136,139-143,147-156,160,163,166,168-172,177-184,187-190,194-197,199,200,202,204,206-209,211,213,214,217,218-223,229,231,234,238,245,246,251-256" --pdb 1 --ncbi_tag "83332" --output_path "example.json"
cd ..
```

Then run the generation code as follows. Please make sure the ncbi tag in the generated example is the same as the one in generate_new_example.sh:

```ruby
bash generate_new_example.sh
```

<h2>Evaluation</h2>

WE provide the pdb to enzyme class (EC) category mapping at [PDB_to_EC_Mapping](https://drive.google.com/file/d/1g5lI2jFKPe1m6U8eu4Onsw7mRkFAA4IT/view?usp=sharing). By using this mapping data, you can gather the results for each enzyme class.

To prepare the data for calculating ESP scores, follow the guidance below:

```ruby
python evaluation/merge
python evaluation/prepare_esp_evaluation_pretrain.py
python evaluation/prepare_esp_evaluation_finetune.py
```

The format for ESP evaluation is (Protein_Sequence Substrate_Representation) for each test case.

The evaluation code for ESP score is developed by Alexander Kroll, which can be found at [link](https://github.com/AlexanderKroll/ESP_prediction_function/tree/main)

<h3>Expected Results for the Pretrained ProteinNet</h3>

![image](./Full_In_Silico_Results_v6.png)

