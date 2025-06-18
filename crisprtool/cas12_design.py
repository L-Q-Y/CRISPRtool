import requests
import pandas as pd
import numpy as np
from functools import reduce
from operator import add
import cyvcf2
import parasail
import re

from model import (
    Cas12_BiLSTM,
    Cas12_SimpleRNN,
    Cas12_MultiHeadAttention,
    Cas12_Transformer,
    Seq_deepCpf1
)


# map model‐name → constructor
MODEL_MAP = {
    'Cas12_BiLSTM':           Cas12_BiLSTM,
    'Cas12_SimpleRNN':        Cas12_SimpleRNN,
    'Cas12_MultiHeadAttention': Cas12_MultiHeadAttention,
    'Cas12_Transformer':      Cas12_Transformer,
    'Seq_deepCpf1':            Seq_deepCpf1
}

WEIGHTS_MAP = {
            'Seq_deepCpf1':     'saved_models/Cas12/Seq_deepCpf1_weights.keras',
            'Cas12_BiLSTM':    "saved_models/Cas12/BiLSTM_Cpf1_weights.keras",
            'Cas12_SimpleRNN': "saved_models/Cas12/SimpleRNN_Cpf1_weights.keras",
            'Cas12_MultiHeadAttention': "saved_models/Cas12/MultiHeadAttention_Cpf1_weights.keras",
            'Cas12_Transformer':       "saved_models/Cas12/Transformer_Cpf1_weights.keras"
        }

# sequence map
ntmap = {'A': (1, 0, 0, 0),
         'C': (0, 1, 0, 0),
         'G': (0, 0, 1, 0),
         'T': (0, 0, 0, 1)
         }


def _load_mutations(mut_path):
    vcf_reader = cyvcf2.VCF(mut_path)
    return vcf_reader


def get_seqcode(seq):
    return np.array(reduce(add, map(lambda c: ntmap[c], seq.upper()))).reshape((1, len(seq), -1))


def fetch_ensembl_transcripts(gene_symbol):
    url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_symbol}?expand=1;content-type=application/json"
    response = requests.get(url)
    if response.status_code == 200:
        gene_data = response.json()
        if 'Transcript' in gene_data:
            return gene_data['Transcript']
        else:
            print("No transcripts found for gene:", gene_symbol)
            return None
    else:
        print(f"Error fetching gene data from Ensembl: {response.text}")
        return None


def fetch_ensembl_sequence(transcript_id):
    url = f"https://rest.ensembl.org/sequence/id/{transcript_id}?content-type=application/json"
    response = requests.get(url)
    if response.status_code == 200:
        sequence_data = response.json()
        if 'seq' in sequence_data:
            return sequence_data['seq']
        else:
            print("No sequence found for transcript:", transcript_id)
            return None
    else:
        print(f"Error fetching sequence data from Ensembl: {response.text}")
        return None


def find_crispr_targets(sequence, chr, start, end, strand, transcript_id, exon_id, pam="TTTN", target_length=34):
    targets = []
    len_sequence = len(sequence)
    #complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    dnatorna = {'A': 'A', 'T': 'U', 'C': 'C', 'G': 'G'}

    for i in range(len_sequence - target_length + 1):
        target_seq = sequence[i:i + target_length]
        if target_seq[4:7] == 'TTT':
            if strand == -1:
                tar_start = end - i - target_length + 1
                tar_end = end -i
                #seq_in_ref = ''.join([complement[base] for base in target_seq])[::-1]
            else:
                tar_start = start + i
                tar_end = start + i + target_length - 1
                #seq_in_ref = target_seq
            gRNA = ''.join([dnatorna[base] for base in target_seq[8:28]])
            targets.append([target_seq, gRNA, chr, str(tar_start), str(tar_end), str(strand), transcript_id, exon_id])
            #targets.append([target_seq, gRNA, chr, str(tar_start), str(tar_end), str(strand), transcript_id, exon_id, seq_in_ref])
    return targets


def format_prediction_output(targets, model_name):
    # Loading weights for the model
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown Cas9 model: {model_name}")
    Constructor = MODEL_MAP[model_name]
    input_shape = (34, 4)
    model = Constructor(input_shape=input_shape)
    model_path = WEIGHTS_MAP[model_name]
    model.load_weights(model_path)

    formatted_data = []
    for target in targets:
        # Predict
        encoded_seq = get_seqcode(target[0])
        prediction = float(list(model.predict(encoded_seq, verbose=0)[0])[0])
        if prediction > 100:
            prediction = 100

        # Format output
        gRNA = target[1]
        chr = target[2]
        start = target[3]
        end = target[4]
        strand = target[5]
        transcript_id = target[6]
        exon_id = target[7]
        #seq_in_ref = target[8]
        #formatted_data.append([chr, start, end, strand, transcript_id, exon_id, target[0], gRNA, seq_in_ref, prediction])
        formatted_data.append([chr, start, end, strand, transcript_id, exon_id, target[0], gRNA, prediction])

    return formatted_data


def gRNADesign(gene_symbol, model_name):
    transcripts = fetch_ensembl_transcripts(gene_symbol)
    results = []
    if transcripts:
        for i in range(len(transcripts)):
            Exons = transcripts[i]['Exon']
            transcript_id = transcripts[i]['id']
            for j in range(len(Exons)):
                exon_id = Exons[j]['id']
                gene_sequence = fetch_ensembl_sequence(exon_id)
                if gene_sequence:
                    start = Exons[j]['start']
                    end = Exons[j]['end']
                    strand = Exons[j]['strand']
                    chr = Exons[j]['seq_region_name']
                    targets = find_crispr_targets(gene_sequence, chr, start, end, strand, transcript_id, exon_id)
                    if targets:
                        formatted_data = format_prediction_output(targets, model_name)
                        results.append(formatted_data)

    header = ['Chrom','Start','End','Strand','Transcript','Exon','Target sequence (5\' to 3\')','gRNA','pred_Score']
    output = []
    for result in results:
        for item in result:
            output.append(item)
    sort_output = sorted(output, key=lambda x: x[8], reverse=True)

    # Create a DataFrame from the sorted output
    df = pd.DataFrame(sort_output, columns=header)
    return df


def apply_mutation(ref_sequence, offset, ref, alt):
    """
    Apply a single mutation to the sequence.
    """
    if len(ref) == len(alt) and alt != "*":  # SNV
        mutated_seq = ref_sequence[:offset] + alt + ref_sequence[offset+len(alt):]

    elif len(ref) < len(alt):  # Insertion
        mutated_seq = ref_sequence[:offset] + alt + ref_sequence[offset+1:]

    elif len(ref) == len(alt) and alt == "*":  # Deletion
        mutated_seq = ref_sequence[:offset] + ref_sequence[offset+1:]

    elif len(ref) > len(alt) and alt != "*":  # Deletion
        mutated_seq = ref_sequence[:offset] + alt + ref_sequence[offset+len(ref):]

    elif len(ref) > len(alt) and alt == "*":  # Deletion
        mutated_seq = ref_sequence[:offset] + ref_sequence[offset+len(ref):]

    return mutated_seq


def construct_combinations(sequence, mutations):
    """
    Construct all combinations of mutations.
    mutations is a list of tuples (position, ref, [alts])
    """
    if not mutations:
        return [sequence]

    # Take the first mutation and recursively construct combinations for the rest
    first_mutation = mutations[0]
    rest_mutations = mutations[1:]
    offset, ref, alts = first_mutation

    sequences = []
    for alt in alts:
        mutated_sequence = apply_mutation(sequence, offset, ref, alt)
        sequences.extend(construct_combinations(mutated_sequence, rest_mutations))

    return sequences


def needleman_wunsch_alignment(query_seq, ref_seq):
    """
    Use Needleman-Wunsch alignment to find the maximum alignment position in ref_seq
    Use this position to represent the position of target sequence with mutations
    """
    # Needleman-Wunsch alignment
    alignment = parasail.nw_trace(query_seq, ref_seq, 10, 1, parasail.blosum62)

    # extract CIGAR object
    cigar = alignment.cigar
    cigar_string = cigar.decode.decode("utf-8")

    # record ref_pos
    ref_pos = 0

    matches = re.findall(r'(\d+)([MIDNSHP=X])', cigar_string)
    max_num_before_equal = 0
    max_equal_index = -1
    total_before_max_equal = 0

    for i, (num_str, op) in enumerate(matches):
        num = int(num_str)
        if op == '=':
            if num > max_num_before_equal:
                max_num_before_equal = num
                max_equal_index = i
    total_before_max_equal = sum(int(matches[j][0]) for j in range(max_equal_index))

    ref_pos = total_before_max_equal

    return ref_pos


def find_gRNA_with_mutation(ref_sequence, exon_chr, start, end, strand, transcript_id,
                            exon_id, gene_symbol, vcf_reader, pam="TTTN", target_length=34):
    # initialization
    mutated_sequences = [ref_sequence]

    # find mutations within interested region
    mutations = vcf_reader(f"{exon_chr}:{start}-{end}")
    if mutations:
        # find mutations
        mutation_list = []
        for mutation in mutations:
            offset = mutation.POS - start
            ref = mutation.REF
            alts = mutation.ALT[:-1]
            mutation_list.append((offset, ref, alts))

        # replace reference sequence of mutation
        mutated_sequences = construct_combinations(ref_sequence, mutation_list)

    # find gRNA in ref_sequence or all mutated_sequences
    targets = []
    for seq in mutated_sequences:
        len_sequence = len(seq)
        dnatorna = {'A': 'A', 'T': 'U', 'C': 'C', 'G': 'G'}
        for i in range(len_sequence - target_length + 1):
            target_seq = seq[i:i + target_length]
            if target_seq[4:7] == 'TTT':
                pos = ref_sequence.find(target_seq)
                if pos != -1:
                    is_mut = False
                    if strand == -1:
                        tar_start = end - pos - target_length + 1
                    else:
                        tar_start = start + pos
                else:
                    is_mut = True
                    nw_pos = needleman_wunsch_alignment(target_seq, ref_sequence)
                    if strand == -1:
                        tar_start = str(end - nw_pos - target_length + 1) + '*'
                    else:
                        tar_start = str(start + nw_pos) + '*'
                gRNA = ''.join([dnatorna[base] for base in target_seq[8:28]])
                targets.append([target_seq, gRNA, exon_chr, str(strand), str(tar_start), transcript_id, exon_id, gene_symbol, is_mut])

    # filter duplicated targets
    unique_targets_set = set(tuple(element) for element in targets)
    unique_targets = [list(element) for element in unique_targets_set]

    return unique_targets


def format_prediction_output_with_mutation(targets, model_name):
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown Cas9 model: {model_name}")
    Constructor = MODEL_MAP[model_name]
    input_shape = (34, 4)
    model = Constructor(input_shape=input_shape)
    model_path = WEIGHTS_MAP[model_name]
    model.load_weights(model_path)

    formatted_data = []

    for target in targets:
        # Encode the gRNA sequence
        encoded_seq = get_seqcode(target[0])


        # Predict on-target efficiency using the model
        prediction = float(list(model.predict(encoded_seq, verbose=0)[0])[0])
        if prediction > 100:
            prediction = 100

        # Format output
        gRNA = target[1]
        exon_chr = target[2]
        strand = target[3]
        tar_start = target[4]
        transcript_id = target[5]
        exon_id = target[6]
        gene_symbol = target[7]
        is_mut = target[8]
        formatted_data.append([gene_symbol, exon_chr, strand, tar_start, transcript_id,
                               exon_id, target[0], gRNA, prediction, is_mut])

    return formatted_data


def gRNADesign_mutation(gene_symbol, model_name, vcf_reader):
    results = []

    transcripts = fetch_ensembl_transcripts(gene_symbol)
    if transcripts:
        for transcript in transcripts:
            Exons = transcript['Exon']
            transcript_id = transcript['id']

            for Exon in Exons:
                exon_id = Exon['id']
                exon_chr = Exon['seq_region_name']
                start = Exon['start']
                end = Exon['end']
                strand = Exon['strand']
                gene_sequence = fetch_ensembl_sequence(exon_id) # reference exon sequence

                if gene_sequence:
                    targets = find_gRNA_with_mutation(gene_sequence, exon_chr, start, end, strand,
                                                      transcript_id, exon_id, gene_symbol, vcf_reader)
                    if targets:
                        # Predict on-target efficiency for each gRNA site
                        formatted_data = format_prediction_output_with_mutation(targets, model_name)
                        results.append(formatted_data)
    header = ['Gene','Chrom','Strand','Start','Transcript','Exon','Target sequence (5\' to 3\')','gRNA','pred_Score','Is_mutation']
    output = []
    for result in results:
        for item in result:
            output.append(item)
    sort_output = sorted(output, key=lambda x: x[8], reverse=True)

    df = pd.DataFrame(sort_output, columns=header)
    return df



def design_sgrnas_cas12(gene_symbol, model_name, use_mutation):
    if use_mutation:
        mut_path = 'data/MDAMB231_mut/SRR25934512.filter.snvs.indels.vcf.gz'
        vcf_reader = _load_mutations(mut_path)
        result_df = gRNADesign_mutation(gene_symbol, model_name, vcf_reader)
    else:
        result_df = gRNADesign(gene_symbol, model_name)
    return result_df













































