import argparse

def parameter_reading():
    parser = argparse.ArgumentParser(
        description="Configuration parameters for training the ICL-Equalizer."
    )

    # --------------------------------------------------------------- #
    # Transformer architecture hyperparameters
    # --------------------------------------------------------------- #
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension of the Transformer input.')
    parser.add_argument('--num_head', type=int, default=8,
                        help='Number of attention heads.')
    parser.add_argument('--num_layer', type=int, default=8,
                        help='Number of Transformer layers.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate used inside the Transformer.')

    # --------------------------------------------------------------- #
    # Data / modulation configuration
    # --------------------------------------------------------------- #
    parser.add_argument('--prompt_seq_length', type=int, default=31,
                        help='Sequence length of the transmitted symbols.')
    parser.add_argument('--num_ant', type=int, default=2,
                        help='Number of antennas in the MIMO system.')

    parser.add_argument('--modulation', default='4QAM',
                        help="Modulation type. Options: '4QAM', '16QAM', '64QAM', 'BPSK', '2PSK'.")
    parser.add_argument('--modu_num', type=int, default=4,
                        help='Number of constellation points (will be overwritten by joint constellation).')

    # --------------------------------------------------------------- #
    # Training hyperparameters
    # --------------------------------------------------------------- #
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size used for training.')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Total number of training epochs.')
    parser.add_argument('--training_steps', type=int, default=3000,
                        help='Training steps (legacy parameter).')

    # --------------------------------------------------------------- #
    # Channel / SNR configuration
    # --------------------------------------------------------------- #
    parser.add_argument('--SNR_dB_min', type=float, default=20,
                        help='Minimum SNR value in dB.')
    parser.add_argument('--SNR_dB_max', type=float, default=20,
                        help='Maximum SNR value in dB.')

    # --------------------------------------------------------------- #
    # Decision-Feedback training configuration
    # --------------------------------------------------------------- #
    parser.add_argument('--train_pilot_len', type=int, default=1,
                        help='Length of pilot symbols provided to DFE training.')
    parser.add_argument('--DFE_TRAIN', type=bool, default=True,
                        help='Enable two-phase training: ICL → DEFINED.')
    parser.add_argument('--DFE_epoch', type=int, default=1000,
                        help='Epoch at which training switches from ICL to DEFINED.')
    parser.add_argument('--loss_weight', type=float, default=0.7,
                        help='Weight for DEFINED loss: loss = w*loss1 + (1-w)*loss2.')

    # --------------------------------------------------------------- #
    # Adaptive DFE switching
    # --------------------------------------------------------------- #
    parser.add_argument('--adaptive_dfe', action='store_true', default=False,
                        help=(
                            'Detect ICL plateau automatically and switch to DFE '
                            'fine-tuning instead of switching at a fixed DFE_epoch. '
                            'DFE_epoch still acts as a hard fallback upper bound.'
                        ))
    parser.add_argument('--dfe_patience', type=int, default=10,
                        help=(
                            'Number of consecutive validation checks (each log_every=10 '
                            'epochs) with less than dfe_min_delta relative improvement '
                            'before the switch to DFE is triggered. Default: 10 (=100 epochs).'
                        ))
    parser.add_argument('--dfe_min_delta', type=float, default=5e-4,
                        help='Minimum relative SER improvement to reset the plateau counter.')
    parser.add_argument('--dfe_min_epochs', type=int, default=1000,
                        help='Minimum ICL epochs that must pass before plateau detection begins.')

    # --------------------------------------------------------------- #
    # Curriculum learning
    # --------------------------------------------------------------- #
    parser.add_argument('--curriculum', action='store_true', default=False,
                        help=(
                            'Progressively increase context sequence length during '
                            'ICL pre-training, starting from curr_start_len and '
                            'growing by curr_step_size every curr_step_epochs epochs.'
                        ))
    parser.add_argument('--curr_start_len', type=int, default=4,
                        help='Initial context length at the start of curriculum training.')
    parser.add_argument('--curr_step_size', type=int, default=3,
                        help='Number of positions added to context length per curriculum step.')
    parser.add_argument('--curr_step_epochs', type=int, default=150,
                        help='Number of training epochs between curriculum length increases.')

    # --------------------------------------------------------------- #
    # Miscellaneous
    # --------------------------------------------------------------- #
    parser.add_argument('--model_type', default='GPT2',
                        help='Model type (kept for compatibility).')

    args = parser.parse_args()
    return args