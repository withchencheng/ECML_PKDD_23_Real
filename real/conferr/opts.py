

def triplet_opts(parser):
    parser.add_argument("--margin", type=float, default=0.4, help="Batch size.")                                                                               
    parser.add_argument("--triple_emb_in_size", type=int, default=768, help="[CLS] size.")                                                                               
    parser.add_argument("--triple_emb_out_size", type=int, default=64, help="triplet net embedding out size.")                                                                               
    parser.add_argument("--distance_type", type=str, default='C', help="Cosine or Entropy.")                                                                                  
    parser.add_argument("--trainer", default=1, type=int, help="Using triplet loss(1) or not(0).")