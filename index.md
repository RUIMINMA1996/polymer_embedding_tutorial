## A Tutorial of Quantifying Polymer's Structure-Property Relationships based on Polymer Embedding

### RUIMIN MA

Numerically representing the physical objects is the first and foremost step for machine learing study in them. Materials are among those physical objects and the field that uses machine learning to dig out the relationship between material structures and properties is called material informatics. Today, I am going to write a simple tutorial on how to use polymer embedding for quantifying structure-property relationships in polymers. 

The method for obtaining polymer embedding is similar to the one used for obtaining word embedding. 

### Import Required Packages
```markdown
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools

import numpy as np 

from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec 

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, KFold

from gensim.models import word2vec

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
```

### Load the polymer2vec model and use it to convert a polymer into corresponding polymer embedding

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

### Choose a machine learning model and train it in leave-one-out cross-validation
```markdown
polymer_embedding_model = word2vec.Word2Vec.load('../data/POLYINFO_PI1M.pkl')

sentences = list()
smiles = ['*CCCCCCCCCCCCCOC(=O)CCC(=O)N*', 
           *CCCCCCCCCOC(=O)CCCCCCCC(*)OC(=O)c1ccccc1,
           *CC(C)CCC(*)OC1C(=O)OCC1(C)C]
for i in range(len(smiles)):
    sentence = MolSentence(mol2alt_sentence(Chem.MolFromSmiles(smiles[i], 1))
    sentences.append(sentence)
polymer_embeddings = [DfVec(x) for x in sentences2vec(sentence, polymer_embedding_model, unseen='UNK')]
```

