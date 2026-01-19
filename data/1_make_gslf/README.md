OpenPOM: https://github.com/BioMachineLearning/openpom.git
- `1_goodcents_dataset_curation.ipynb`
- `2_leffingwell_dataset_curation.ipynb`
- `3_merge_datasets.ipynb`

&rarr; `curated_GS_LF_merged_4986.csv` with 152 labels
```python
required_desc = [
    'acidic', 'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal', 
    'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy', 'bergamot', 
    'berry', 'bitter', 'black currant', 'brandy', 'bready', 'brown', 'burnt', 'buttery', 
    'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery', 'chamomile', 'cheesy', 
    'chemical', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean', 'clove', 'cocoa', 
    'coconut', 'coffee', 'cognac', 'cooked', 'cooling', 'cortex', 'coumarinic', 'creamy', 
    'cucumber', 'dairy', 'dry', 'earthy', 'estery', 'ethereal', 'fatty', 'fermented', 
    'fishy', 'floral', 'fresh', 'fruit skin', 'fruity', 'fungal', 'fusel', 'garlic', 
    'gassy', 'geranium', 'grape', 'grapefruit', 'grassy', 'green', 'hawthorn', 'hay', 
    'hazelnut', 'herbal', 'honey', 'hyacinth', 'jammy', 'jasmin', 'juicy', 'ketonic', 
    'lactonic', 'lavender', 'leafy', 'leathery', 'lemon', 'licorice', 'lily', 'malty', 
    'marine', 'meaty', 'medicinal', 'melon', 'mentholic', 'metallic', 'milky', 'mint', 
    'mossy', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty', 'odorless', 'oily', 
    'onion', 'orange', 'orangeflower', 'orris', 'ozone', 'peach', 'pear', 'phenolic', 'pine', 
    'pineapple', 'plum', 'popcorn', 'potato', 'powdery', 'pungent', 'radish', 'raspberry', 
    'ripe', 'roasted', 'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy', 
    'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet', 'tea', 'terpenic', 
    'thujonic', 'tobacco', 'tomato', 'tonka', 'tropical', 'vanilla', 'vegetable', 'vetiver', 
    'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
]
```

POMMix: https://github.com/chemcognition-lab/pom-mix.git
- `4_dataset_cleaner.ipynb`

&rarr; For a fair comparison, we subset the 152-label dataset to the original 4,983 SMILES (`curated_GS_LF_merged_4983.csv`) and applied the same POMMix preprocessing; the cleaned set likewise converged to 4,814 SMILES, matching the original POMMix result.