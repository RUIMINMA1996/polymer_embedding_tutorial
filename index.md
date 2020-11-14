## A Tutorial of Quantifying Polymer's Structure-Property Relationships based on Polymer Embedding

### RUIMIN MA

Numerically representing the physical objects is the first and foremost step for machine learing study in them. Materials are among those physical objects and the field that uses machine learning to dig out the relationship between material structures and properties is called material informatics. Today, I am going to write a simple tutorial on how to use polymer embedding to fast quantify structure-property relationships in polymers. 

The method used for obtaining polymer embedding is called [Skip-gram](https://towardsdatascience.com/skip-gram-nlp-context-words-prediction-algorithm-5bbf34f84e0c). Specifically, a polymer is decomposed into a sequence of substructures following the [Morgan algorithm](https://pubs.acs.org/doi/abs/10.1021/ci100050t), and then a target substructure in this sequence of substructures is picked up to predict its context substructures via a single-layer neural network. Theoretically, each substructure in the sequence is used as the target substructure for once per training epoch. When the training is done, the weights of the neural network are treated as polymer embedding.

Both the Morgan fingerprint and polymer embedding can be used as polymer representation for fast quantifying structure-property relationships, however, polymer embedding is proved to be a more informative representation. Detailed comparison between those two can be found at [Evaluating Polymer Representations via Quantifying Structure–Property Relationships](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00358).

Each polymer is first physically represented as a p-SMILES, which is a '*'-decorated SMILES representation of its monomer. p-SMILES is proved to have all the atomic and bonding information of a polymer. Detailed discussion can be found at [PI1M: A Benchmark Database for Polymer Informatics](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00726).

### Import Required Packages
```markdown
from rdkit import Chem
import numpy as np 
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec 

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut

from gensim.models import word2vec

from tqdm import tqdm

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
```

rdkit is a well-known cheminformatics packge for manipulating molecules, and it is used here to convert the p-SMILES into the corresponding molecular object that can be decomposed into sequence of substructures by mol2vec. gensim is a well-known natural language processing package that contains many language models, like word2vec. It is used for loading the pretrained polymer embedding model here. sklearn is a machine learning platform, where machine learning model can be fast built from it.

### Load the pretrained polymer embedding and use it to featurize polymers

```markdown
polymer_embedding_model = word2vec.Word2Vec.load('../data/POLYINFO_PI1M.pkl')

sentences = list()
smiles = ['*CCCCCCCCCCCCCOC(=O)CCC(=O)N*', 
           '*CCCCCCCCCOC(=O)CCCCCCCC(*)OC(=O)c1ccccc1',
           '*CC(C)CCC(*)OC1C(=O)OCC1(C)C']
           
for i in range(len(smiles)):
    sentence = MolSentence(mol2alt_sentence(Chem.MolFromSmiles(smiles[i], 1))
    sentences.append(sentence)
    
polymer_embeddings = [DfVec(x) for x in sentences2vec(sentence, polymer_embedding_model, unseen='UNK')]
```
steps:
           -load the pretrained [polymer embedding model](https://drive.google.com/file/d/1_lfHWOLcV3VX37wN7kuBDFTAn6EYPQeD/view?usp=sharing)
           -a few p-SMILES are listed here for demonstration, and millions of them can be read from [PI1M](https://github.com/RUIMINMA1996/PI1M).
           -convert p-SMILES into molecular object and decomposed it into sequence of substructures
           -convert substructures into polymer embeddings and add them up

### Choose a machine learning model and train it in leave-one-out cross-validation
```markdown
X = np.array([x.vec.tolist() for x in polymer_embeddings])
y = np.array([a, b, c]) # a, b, c can be the corresponding property values

MAEs = []
predictions = list()
ground_truths = list()

rng = RandomForestRegressor(n_estimators=100, random_state=0, oob_score=False, n_jobs=-1)
loo = LeaveOneOut()

for train_index, test_index in tqdm(loo.split(X)):
    rng.fit(X[train_index], y[train_index])
    prediction = rng.predict(X[test_index])
    ground_truth = y[test_index]

    predictions.append(prediction[0])
    ground_truths.append(ground_truth[0])
    
    MAE = abs(prediction[0] - ground_truth[0])
    MAEs.append(MAE)
```
steps:
           -built inputs (X, y) for machine learing
           -create some lists for recording outputs
           -initialize the machine learning model and data-splitting method
           -train the model via leave-one-out cross-validation

### Summary
Quantitative strucure-property relationships is the key of material informatics. Hope this simple tutorial will help more and more polymer researchers in finding and designing polymers. If you are interested and want to know more details, please let me know. I will try to get back to you.


