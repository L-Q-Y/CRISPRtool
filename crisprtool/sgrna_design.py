import argparse
from cas9_design import  design_sgrnas_cas9
from cas12_design import design_sgrnas_cas12


def parse_args():
    p = argparse.ArgumentParser(
        description="design sgRNAs for any genes"
    )
    p.add_argument('--group',
                   choices=['cas9','cas12'],
                   required=True,
                   help="which nuclease family to evaluate & design for")
    p.add_argument('--model',
                   required=True,
                   help="model want to use")
    p.add_argument('--gene',
                   required=True,
                   help="gene name to design for")
    p.add_argument('--use-mutation',
                   action='store_true',
                   help="if set, include mutation info from /data/MDAMB231_mut")
    p.add_argument('--save-csv',
                   metavar='FILE',
                   help="if set, save full results here; else prints top 10")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    if args.group == 'cas9':
        df_sg = design_sgrnas_cas9(
            gene_symbol        = args.gene,
            model_name   = args.model,
            use_mutation = args.use_mutation
        )
    else:
        df_sg = design_sgrnas_cas12(
            gene_symbol        = args.gene,
            model_name   = args.model,
            use_mutation = args.use_mutation
        )

    if args.save_csv:
        df_sg.to_csv(args.save_csv, index=False)
        print(f"All {len(df_sg)} candidates saved to {args.save_csv}")
    else:
        print("Top 10 sgRNA candidates:")
        print(df_sg.head(10).to_string(index=False))


if __name__ == '__main__':
    main()






















