import os

def filter_lines(text):
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        tks = line.split('\t')
        if len(tks) == 6:
            if tks[4] == "Disease":
                tks[4] = "DiseaseOrPhenotypicFeature"
            elif tks[4] == "Gene": 
                tks[4] = "GeneOrGeneProduct"
            elif tks[4] == "Chemical":
                tks[4] = "ChemicalEntity"
            elif tks[4] == "Species":
                tks[4] = "OrganismTaxon"
            elif tks[4] == "ProteinMutation" or tks[4] == "DNAMutation" or tks[4] == "Mutation" or tks[4] == "Variant" or tks[4] == "SNP":
                tks[4] = "SequenceVariant"
            if line[-1] == "null":
                continue
            elif tks[4] == 'SequenceVariant':
                id = ''
                for _id in tks[5].rstrip().split(';'):
                    if _id.startswith('RS#:'):
                        id = 'rs' + _id[4:]
                if id == '':
                    for _id in tks[5].rstrip().split(';'):
                        if _id.startswith('tmVar:'):
                            id = _id[6:]
                tks[5] = id
            elif tks[4] == 'DNAAcidChange' or tks[4] == 'ProteinAcidChange':
                continue
            tks[-1] = tks[-1].replace(';', ',').replace("MESH:", "")
            tks[-1] = tks[-1].replace('CVCL:', 'CVCL_')
            line = '\t'.join(tks)
            
        elif len(tks) == 4:
            continue
        filtered_lines.append(line)
    return '\n'.join(filtered_lines)

in_pubtator_file  = 'samples/sample.pubtator'
out_pubtator_file = 'samples/sample.biored.pubtator'

with open(out_pubtator_file, 'w') as write_file:
    with open(in_pubtator_file, 'r') as current_file:
        content = current_file.read()
        content = filter_lines(content)
        write_file.write(content)
