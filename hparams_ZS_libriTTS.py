# adapted from https://github.com/NVIDIA/tacotron2/blob/master/hparams.py
# add some modifications

import tensorflow as tf
from text import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=828,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers= ['language_embedding','speaker_embedding', 'speaker_encoder'], #['embedding.weight','language_embedding.weight', 'speaker_embedding.weight'],
        loss_type=1,  # 0:Gate+Mel 1:Gate+Mel+SEmb

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files=  't2_datainfo/LibriTTS_Train360_filelists.txt',
        validation_files = 't2_datainfo/LibriTTS_Dev_filelists.txt',
        text_cleaners= ['english_cleaners'], #['indonesia_cleaners'], ['english_cleaners'],
        file_path = '',
        p_arpabet=1.0, # proportion/probability of using phoneme level, e.g, 0:grapheme level, 0.5: grapheme and phoneme fivty fivty, 1.0: phoneme level
        cmudict_path="data/cmu_dictionary",

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate= 22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000, #ori 1000
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        # Language Embedding parameters
        n_languages=1,
        language_embedding_dim=8,
        with_language_embedding=True,

        # Speaker Embedding parameters
        n_speakers=1230,
        speaker_embedding_dim=128,
        with_speaker_embedding=True,
        speaker_embedding_type=0,  # 0: at encoder, 1: at encoder & decoder-prenet
        speaker_embedding_file= "t2_datainfo/LibriTTS_embed_dv_libriTTS_m80sr22.npy", #use pre-trained speaker embedding
        speaker_ids_file="t2_datainfo/LibriTTS_speaker_ids.npy", #use predefined speaker-id
        use_speaker_encoder=2, #0: simple nn embedding, 1: use external speaker encoder embedding, 2: use pretrained d-vector speaker encoder
        SE_model_path= "pretrained/speaker_encoder/se_libriTTS_m80sr22/best_model_50k1.pth.tar",
        SE_config_path = "pretrained/speaker_encoder/se_libriTTS_m80sr22/config.json",
        embedded_speaker_path= "../dataset/speaker_embeddings/dv_libriTTS_m80sr22",

        # d-vector speaker encoder parameters
        se_input_dim = 80,
        se_proj_dim = 128 ,
        se_lstm_dim = 384,
        se_num_lstm_layers = 3,

        #Waveglow
        waveglow_path = '',

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=16,
        gradient_accum = 4, # batch_size*gradient_accum = the real batch_size
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.compat.v1.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.compat.v1.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
