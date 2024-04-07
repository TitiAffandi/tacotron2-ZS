# adapted from https://github.com/NVIDIA/tacotron2/blob/master/data_utils.py
# add some modifications

import os
import random
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cmudict


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs,language_ids and speaker_ids
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams, language_ids=None, speaker_ids=None, shuffle=True):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.file_path = hparams.file_path
        self.with_language_embedding = hparams.with_language_embedding
        self.with_speaker_embedding = hparams.with_speaker_embedding
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

        self.p_arpabet = hparams.p_arpabet
        self.cmudict = None
        if hparams.cmudict_path is not None:
            self.cmudict = cmudict.CMUDict(hparams.cmudict_path)

        self.language_ids = language_ids
        if language_ids is None:
            self.language_ids = self.create_language_lookup_table(self.audiopaths_and_text)

        self.use_speaker_encoder = hparams.use_speaker_encoder
        self.embedded_speaker_path = hparams.embedded_speaker_path
        self.speaker_ids = speaker_ids
        self.speaker_ids_file = hparams.speaker_ids_file
        if speaker_ids is None:
            self.speaker_ids = self.create_speaker_lookup_table(self.audiopaths_and_text)

        if shuffle:
            random.seed(1234)
            random.shuffle(self.audiopaths_and_text)

    def create_language_lookup_table(self, audiopaths_and_text):
        language_ids = np.sort(np.unique([x[2] for x in audiopaths_and_text]))
        #d = {int(language_ids[i]): i for i in range(len(language_ids))}
        d = {language_ids[i]: i for i in range(len(language_ids))}
        return d

    def create_speaker_lookup_table(self, audiopaths_and_text):
        if self.speaker_ids_file != "":
            speaker_ids = np.load(self.speaker_ids_file)
        else:
            speaker_ids = np.sort(np.unique([x[3] for x in audiopaths_and_text]))
        d = {speaker_ids[i]: i for i in range(len(speaker_ids))}
        #print("\n[TextMelLoader] speaker_lookup_table file:", d)
        #speaker_idst = np.sort(np.unique([x[3] for x in audiopaths_and_text]))
        #dt = {speaker_idst[i]: i for i in range(len(speaker_idst))}
        #print("\n[TextMelLoader] speaker_lookup_table train:", dt)
        return d

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        if self.file_path != "":
            audiopath = os.path.join( self.file_path, audiopath)
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            #print("SR {} STFT SR {} ".format(sampling_rate, self.stft.sampling_rate))
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        #text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        text_norm = torch.IntTensor(
            text_to_sequence(text, self.text_cleaners, self.cmudict, self.p_arpabet))
        return text_norm

    def get_data(self, audiopath_and_text):
        text, mel = self.get_mel_text_pair(audiopath_and_text)

        language_id = self.get_language_id(audiopath_and_text[2])

        speaker_id = self.get_speaker_id(audiopath_and_text[3])
        if self.use_speaker_encoder == 1: # use pre-computed embedding
            speaker_embedd = self.get_speaker_embedding(audiopath_and_text[3], audiopath_and_text[0])
        else:
            speaker_embedd = speaker_id[:, None]

        #print("\n[TextMelLoader] text:", text.shape)
        #print("[TextMelLoader] mel:",mel.shape)
        #print("[TextMelLoader] s_id:",speaker_id.shape)
        #print("[TextMelLoader] s_embed:",speaker_embedd.shape)
        #print(speaker_embedd)

        return (text, mel, language_id, speaker_id, speaker_embedd)

    def get_language_id(self, language_id):
        #return torch.IntTensor([self.language_ids[int(language_id)]])
        return torch.IntTensor([self.language_ids[language_id]])

    def get_speaker_id(self, speaker_id):
        return torch.IntTensor([self.speaker_ids[speaker_id]])

    def get_speaker_embedding(self, speaker, audiopath):
        audiopath = os.path.join(self.embedded_speaker_path, speaker,os.path.basename(audiopath).replace(".wav", ".npy"))
        speaker_embedd = np.load(audiopath)
        speaker_embedd = torch.from_numpy(speaker_embedd)
        speaker_embedd = speaker_embedd.squeeze(0)
        return speaker_embedd

    def __getitem__(self, index):
        #return self.get_mel_text_pair(self.audiopaths_and_text[index])
        return self.get_data(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        #speaker embedd size
        embedd_size = batch[0][4].size(0)

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        language_ids = torch.LongTensor(len(batch))
        speaker_ids = torch.LongTensor(len(batch))
        speaker_embedd = torch.FloatTensor(len(batch), embedd_size)
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            language_ids[i] = batch[ids_sorted_decreasing[i]][2]
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][3]
            speaker_embedd[i] = batch[ids_sorted_decreasing[i]][4]

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, language_ids, speaker_ids, speaker_embedd
