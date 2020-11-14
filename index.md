## A Tutorial of Quantifying Polymer's Structure-Property Relationships based on Polymer Embedding

Numerically representing the physical objects is the first and foremost step for machine learing study in them. Materials are among those physical objects and the field that uses machine learning to dig out the relationship between material structures and properties is called material informatics. Today, I am going to write a simple tutorial on how to use polymer embedding for quantifying structure-property relationships in polymers. 

The method for obtaining polymer embedding is similar to the one used for obtaining word embedding. 

### Import Required Packages
```markdown
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools

import scipy

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg     

from gensim.models import word2vec
from sklearn.manifold import TSNE 

from tqdm import tqdm 

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
```

### Load the polymer2vec model and use it to convert a polymer into corresponding polymer embedding


