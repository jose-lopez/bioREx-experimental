import pandas as pd
import re
import logging
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model and tokenizer
model_path = "pretrained_model"  # Update this path to your model's location
model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define the label map
id2label = {
    0: 'None',
    1: 'Association',
    2: 'Bind',
    3: 'Comparison',
    4: 'Conversion',
    5: 'Cotreatment',
    6: 'Drug_Interaction',
    7: 'Negative_Correlation',
    8: 'Positive_Correlation',
    9: 'None-CID',
    10: 'CID',
    11: 'None-PPIm',
    12: 'PPIm',
    13: 'None-AIMED',
    14: 'None-DDI',
    15: 'None-BC7',
    16: 'None-phargkb',
    17: 'None-GDA',
    18: 'None-DISGENET',
    19: 'None-EMU_BC',
    20: 'None-EMU_PC',
    21: 'None-HPRD50',
    22: 'None-PHARMGKB',
    23: 'ACTIVATOR',
    24: 'AGONIST',
    25: 'AGONIST-ACTIVATOR',
    26: 'AGONIST-INHIBITOR',
    27: 'ANTAGONIST',
    28: 'DIRECT-REGULATOR',
    29: 'INDIRECT-DOWNREGULATOR',
    30: 'INDIRECT-UPREGULATOR',
    31: 'INHIBITOR',
    32: 'PART-OF',
    33: 'PRODUCT-OF',
    34: 'SUBSTRATE',
    35: 'SUBSTRATE_PRODUCT-OF',
    36: 'mechanism',
    37: 'int',
    38: 'effect',
    39: 'advise',
    40: 'AIMED-Association',
    41: 'HPRD-Association',
    42: 'EUADR-Association',
    43: 'None-EUADR',
    44: 'Indirect_conversion',
    45: 'Non_conversion'
}

# Provided PubTator data
pubtator_data = """
10072520|t|Lymphocyte activation gene-3, a MHC class II ligand expressed on activated T cells, stimulates TNF-alpha and IL-12 production by monocytes and dendritic cells.
10072520|a|Lymphocyte activation gene-3 (LAG-3) is an MHC class II ligand structurally and genetically related to CD4. Although its expression is restricted to activated T cells and NK cells, the functions of LAG-3 remain to be elucidated. Here, we report on the expression and function of LAG-3 on proinflammatory bystander T cells that are activated in the absence of TCR engagement. LAG-3 is expressed at high levels on human T cells cocultured with autologous monocytes and IL-2 and synergizes with the low levels of CD40 ligand (CD40L) expressed on these cells to trigger TNF-alpha and IL-12 production by monocytes. Indeed, anti-LAG-3 mAb inhibits both IL-12 and IFN-gamma production in IL-2-stimulated cocultures of T cells and autologous monocytes. Soluble LAG-3Ig fusion protein markedly enhances IL-12 production by monocytes stimulated with infra-optimal concentrations of sCD40L, whereas it directly stimulates monocyte-derived dendritic cells (DC) for the production of TNF-alpha and IL-12, unravelling an enhanced responsiveness to MHC class II engagemenent in DC as compared with activated monocytes. Thus similar to CD40L, LAG-3 may be involved in the proinflammatory activity of cytokine-activated bystander T cells and most importantly it may directly activate DC.
10072520\t0\t28\tLymphocyte activation gene-3\tGene\t3902
10072520\t95\t104\tTNF-alpha\tGene\t7124
10072520\t109\t114\tIL-12\tGene\t3593
10072520\t160\t188\tLymphocyte activation gene-3\tGene\t3902
10072520\t190\t195\tLAG-3\tGene\t3902
10072520\t263\t266\tCD4\tGene\t920
10072520\t358\t363\tLAG-3\tGene\t3902
10072520\t439\t444\tLAG-3\tGene\t3902
10072520\t448\t463\tproinflammatory\tDisease\t
10072520\t519\t522\tTCR\tGene\t6962
10072520\t535\t540\tLAG-3\tGene\t3902
10072520\t572\t577\thuman\tSpecies\t9606
10072520\t627\t631\tIL-2\tGene\t3558
10072520\t670\t681\tCD40 ligand\tGene\t959
10072520\t683\t688\tCD40L\tGene\t959
10072520\t726\t735\tTNF-alpha\tGene\t7124
10072520\t740\t745\tIL-12\tGene\t3593
10072520\t784\t789\tLAG-3\tGene\t3902
10072520\t808\t813\tIL-12\tGene\t3593
10072520\t818\t827\tIFN-gamma\tGene\t3458
10072520\t842\t846\tIL-2\tGene\t3558
10072520\t955\t960\tIL-12\tGene\t3593
10072520\t1132\t1141\tTNF-alpha\tGene\t7124
10072520\t1146\t1151\tIL-12\tGene\t3593
10072520\t1281\t1286\tCD40L\tGene\t959
10072520\t1288\t1293\tLAG-3\tGene\t3902
10072520\t1317\t1332\tproinflammatory\tDisease\t
"""

# Parse the PubTator data
sentences = {}
annotations = []
for line in pubtator_data.strip().split('\n'):
    parts = line.split('|')
    if len(parts) > 2:
        doc_id, section, text = parts
        if doc_id not in sentences:
            sentences[doc_id] = {'title': '', 'abstract': ''}
        if section == 't':
            sentences[doc_id]['title'] = text
        elif section == 'a':
            sentences[doc_id]['abstract'] = text
    else:
        parts = line.split('\t')
        if len(parts) > 3:
            doc_id, start, end, entity, entity_type, *rest = parts
            if entity_type == "Gene":
                annotations.append((doc_id, entity, int(start), int(end)))

# Combine title and abstract
combined_sentences = {doc_id: f"{content['title']} {content['abstract']}" for doc_id, content in sentences.items()}


# Function to mark genes in the text
def mark_genes(sentence, gene, tag_type):
    # Ensure the gene is marked correctly
    pattern = re.compile(re.escape(gene), re.IGNORECASE)
    return pattern.sub(f"@{tag_type}$ {gene} @{tag_type}/$", sentence)


# Generate gene pairs for each document and mark the genes
gene_pairs = []
for i, (doc_id, geneA, startA, endA) in enumerate(annotations):
    for j, (doc_id2, geneB, startB, endB) in enumerate(annotations):
        if i < j and doc_id == doc_id2:
            sentence = combined_sentences[doc_id]
            marked_sentence = mark_genes(sentence, geneA, "GeneOrGeneProductSrc")
            marked_sentence = mark_genes(marked_sentence, geneB, "GeneOrGeneProductTgt")
            gene_pairs.append((marked_sentence, geneA, geneB))

# Create DataFrame from gene pairs
df = pd.DataFrame(gene_pairs, columns=["Sentence", "GeneA", "GeneB"])

# Modify the sentences to include the specific format
df['Sentence'] = df.apply(lambda
                              row: f"What is [Litcoin] between @GeneOrGeneProductSrc$ {row['GeneA']} @/GeneOrGeneProductSrc$ and @GeneOrGeneProductTgt$ {row['GeneB']} @/GeneOrGeneProductTgt$ ? [SEP] {row['Sentence']}",
                          axis=1)


def predict_relationship(sentence, geneA, geneB):
    # Combine sentence with GeneA and GeneB
    combined_input = f"{sentence}"

    # Tokenize the input
    inputs = tokenizer(combined_input, return_tensors="tf", truncation=True, padding=True, max_length=512)

    # Make predictions
    outputs = model(inputs)
    predictions = tf.argmax(outputs.logits, axis=1)
    predicted_class = predictions.numpy()[0]

    return predicted_class


# Apply the prediction function to the DataFrame
df['Predicted_Class'] = df.apply(lambda row: predict_relationship(row['Sentence'], row['GeneA'], row['GeneB']), axis=1)
df['Predicted_Label'] = df['Predicted_Class'].map(id2label)

# Output the DataFrame with predictions
print(df)

# Optionally, log the results
logger.info(df.to_string(index=False))